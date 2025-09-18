from typing import Any, Dict, Iterable, Optional
import torch

from .base_task import BaseTask
from utils.poisson import poisson_count_loss, infer_count
from dataset.base_dataset_adapter import BaseDatasetAdapter
from models.base_model_adapter import BaseModelAdapter
import logging


class SpeakerCountTask(BaseTask):
    def __init__(self, *, min_time: Optional[float] = None, max_time: Optional[float] = None):
        super().__init__()
        # not really times but for backwards compatibility with timestamp single lol
        self.min_time = min_time
        self.max_time = max_time

    def _build_conversation_text(self, *, model_processor: Any, ds_adapter: BaseDatasetAdapter, speaker_count: int, eval_mode: bool) -> str:
        prompt = ds_adapter.get_speaker_count_prompt()

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "audio": "PLACEHOLDER AUDIO"},
                ],
            },
        ]

        # Training supervision: include expected JSON as assistant text when not in eval
        if not eval_mode:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"{speaker_count}"},
                    ],
                }
            )

        return model_processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=eval_mode)

    def build_labels(
        self,
        *,
        example: Dict[str, Any],
        ds_adapter: BaseDatasetAdapter,
        model_adapter: BaseModelAdapter,
        eval_mode: bool,
    ) -> Dict[str, Any]:
        audio = ds_adapter.get_audio(example)
        speaker_count = ds_adapter.get_num_speakers(example)

        # Build prompt text via chat template and prepare inputs using the model adapter's processor
        processor = model_adapter.processor
        prompt_text = self._build_conversation_text(
            model_processor=processor, ds_adapter=ds_adapter, speaker_count=speaker_count, eval_mode=eval_mode)

        audio_frames = audio["array"]
        assert int(audio["sampling_rate"]) == model_adapter.sampling_rate

        inputs = processor(
            text=prompt_text,
            audio=audio_frames,
            return_tensors="pt",
            padding=True,
        )

        if eval_mode:
            return inputs

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_features = inputs["input_features"]
        feature_attention_mask = inputs["feature_attention_mask"]

        # Labels aligned to <AUDIO> embeddings (40 ms per embedding for Qwen2.5 Omni)
        labels_size = int((input_ids == model_adapter.audio_id).sum().item())
        labels = torch.zeros(labels_size, device=input_ids.device)

        labels[0] = speaker_count

        # Mask out everything up to and including the assistant token
        label_ids = input_ids.clone()
        assistant_idx = (input_ids == model_adapter.assistant_id).nonzero(
            as_tuple=True)[1][0]
        label_ids[0, : assistant_idx + 1] = -100

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "input_features": input_features[0],
            "feature_attention_mask": feature_attention_mask[0],
            "labels": labels,
            "label_ids": label_ids[0],
        }

    # ----- Evaluation helpers -----
    def evaluate_tokens(
        self,
        *,
        example: Dict[str, Any],
        ds_adapter: BaseDatasetAdapter,
        model: Any,
        error_bound: float = 0.1,
    ) -> Dict[str, Any]:
        logging.info("DONT FORGET TO RETURN NONE IF IT DOESNT WORK")
        speaker_count = ds_adapter.get_num_speakers(example)

        # Build evaluation inputs via build_labels to mirror evaluate.py behavior
        inputs = self.build_labels(
            example=example,
            ds_adapter=ds_adapter,
            model_adapter=model.adapter,
            eval_mode=True,
        )
        inputs = inputs.to(next(model.parameters()).device)

        processor = model.adapter.processor
        with torch.no_grad():
            tokens = model.generate(
                **inputs, eos_token_id=processor.tokenizer.eos_token_id
            )
        generated_tokens = tokens[0][inputs["input_ids"].shape[1]:-1]
        generated_string = processor.tokenizer.decode(generated_tokens)
        logging.info(f"Token prediction: {generated_string}, GT: {speaker_count}")
        try:
            token_pred = int(generated_string)
        except Exception:
            return {"token_correct": 0.0, "parsing_error": 1.0}

        # Metric increments
        abs_err = abs(token_pred - speaker_count)
        metrics: Dict[str, float] = {
            "token_abs_error_sum": abs_err,
            "token_correct": 1.0 if token_pred == speaker_count else 0.0,
        }
        return metrics

    def evaluate_auxiliary_outputs(
        self,
        *,
        example: Dict[str, Any],
        ds_adapter: BaseDatasetAdapter,
        model: Any,
        event: Optional[Dict[str, Any]] = None,
        error_bound: float = 0.1,
    ) -> Dict[str, Any]:
        speaker_count = ds_adapter.get_num_speakers(example)

        # Build evaluation inputs via build_labels to mirror evaluate.py behavior
        inputs = self.build_labels(
            example=example,
            ds_adapter=ds_adapter,
            model_adapter=model.adapter,
            eval_mode=True,
        )
        inputs = inputs.to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model.adapter(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[model.audio_layer]
            audio_hidden_states = hidden_states[inputs["input_ids"]
                                                == model.adapter.audio_id]
            logits = model.linear(audio_hidden_states).squeeze()
            logits = logits.unsqueeze(0)
            if model.poisson_loss:
                aux_pred = infer_count(logits, torch.ones_like(logits)).item()
            else:
                # probs = torch.sigmoid(logits)
                # logging.info(f"Sigmoid probs: {probs}")
                # aux_pred = torch.round(probs.sum()).item()

                prob = logits[:, -1]
                aux_pred = torch.round(prob).item()

        logging.info(f"Auxiliary prediction: {aux_pred}, GT: {speaker_count}")

        # Metric increments
        abs_err = abs(aux_pred - speaker_count)
        metrics: Dict[str, float] = {
            "aux_abs_error_sum": abs_err,
            "aux_correct": 1.0 if aux_pred == speaker_count else 0.0,
        }
        return metrics

    def calculate_loss(self, logits, labels, adapter: BaseModelAdapter, use_poisson_loss: bool, class_weighting: bool) -> torch.Tensor:
        # Get ground truth count with cleanup
        labels_sum = labels.sum()
        gt_count = labels_sum.item()

        if use_poisson_loss:
            logging.info("Poisson loss enabled")
            # Create batched tensors with explicit cleanup
            logits_batched = logits.unsqueeze(0)  # make it look batched
            counts = labels_sum  # reuse the already computed sum
            frame_mask = torch.ones_like(logits_batched)

            gt_count = counts.item()
            pred_count_tensor = infer_count(logits_batched, frame_mask)
            pred_count = pred_count_tensor.item()

            logging.info(f"Labels Ground Truth: {gt_count}")
            logging.info(f"Predicted Count: {pred_count}")

            loss = poisson_count_loss(logits_batched, counts, frame_mask)
            
            # Clean up intermediate tensors
            del logits_batched, frame_mask, pred_count_tensor

            return loss, abs(pred_count - gt_count)
        else:
            logging.info("Bernoulli loss enabled")

            # probs = torch.sigmoid(logits)
            # logging.info(f"Sigmoid probs: {probs}")
            # raw_count = probs.sum()

            raw_count = logits[-1]

            pred_count_tensor = torch.round(raw_count)
            pred_count = pred_count_tensor.item()

            loss = torch.abs(raw_count - labels_sum)

            logging.info(f"Labels Ground Truth: {gt_count}")
            logging.info(f"Predicted Count: {pred_count}")

            # Clean up intermediate tensors
            del labels_sum, pred_count_tensor

            return loss, torch.tensor(abs(pred_count - gt_count)).to(loss.device)

	# [min, max)
    def skip_example(self, example: Dict[str, Any], adapter: BaseModelAdapter) -> bool:
        if self.min_time is not None and adapter.get_num_speakers(example) < self.min_time:
            return True
        if self.max_time is not None and adapter.get_num_speakers(example) >= self.max_time: 
            return True
        return False