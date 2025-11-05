from typing import Any, Dict, Iterable, Optional
import torch

from .base_task import BaseTask
from utils.poisson import poisson_count_loss, infer_count
from dataset.base_dataset_adapter import BaseDatasetAdapter
from models.base_model_adapter import BaseModelAdapter
import logging
from accelerate import PartialState


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
        audio_frames = ds_adapter.get_audio_frames(example)
        speaker_count = ds_adapter.get_num_speakers(example)

        # Build prompt text via chat template and prepare inputs using the model adapter's processor
        processor = model_adapter.processor
        prompt_text = self._build_conversation_text(
            model_processor=processor, ds_adapter=ds_adapter, speaker_count=speaker_count, eval_mode=eval_mode)

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
        audio_labels_size = int((input_ids == model_adapter.audio_id).sum().item())
        audio_labels = torch.zeros(audio_labels_size, device=input_ids.device)

        audio_labels[0] = speaker_count

        # Mask out everything up to and including the assistant token
        labels = input_ids.clone()
        assistant_idx = (input_ids == model_adapter.assistant_id).nonzero(
            as_tuple=True)[1][0]
        labels[0, : assistant_idx + 1] = -100

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "input_features": input_features[0],
            "feature_attention_mask": feature_attention_mask[0],
            "audio_labels": audio_labels,
            "labels": labels[0],
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
            model_adapter=model.model_adapter,
            eval_mode=True,
        )
        inputs = inputs.to(next(model.parameters()).device)

        generated_string = model.generate(**inputs)
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
        gt_counts = ds_adapter.get_num_speakers(example)

        # Build evaluation inputs via build_labels to mirror evaluate.py behavior
        inputs = self.build_labels(
            example=example,
            ds_adapter=ds_adapter,
            model_adapter=model.model_adapter,
            eval_mode=True,
        )
        inputs = inputs.to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model(**inputs, inference=True)
            pred_counts = outputs.auxiliary_prediction.item()

        logging.info(f"Auxiliary prediction: {pred_counts}, GT: {gt_counts}")

        # Metric increments
        abs_err = abs(pred_counts - gt_counts)
        logging.info(f"Absolute error: {abs_err}, Error bound: {error_bound}, Correct: {1.0 if pred_counts == gt_counts else 0.0}")
        metrics: Dict[str, float] = {
            "aux_abs_error_sum": abs_err,
            "aux_correct": 1.0 if pred_counts == gt_counts else 0.0,
        }
        return metrics

    def calculate_loss(self, audio_logits, audio_labels, audio_labels_frame_mask, model_adapter: BaseModelAdapter, use_poisson_loss: bool, class_weighting: bool) -> torch.Tensor:
        gt_counts = audio_labels[:, 0] # (batch,)

        if use_poisson_loss:
            logging.info("Poisson loss enabled")
            loss = poisson_count_loss(audio_logits, gt_counts, audio_labels_frame_mask).mean()
            pred_counts = infer_count(audio_logits, audio_labels_frame_mask)
            pred_abs_diff = torch.abs(pred_counts - gt_counts)

            if PartialState().is_main_process:
                logging.info(f"gt_counts: {gt_counts}, pred_counts: {pred_counts}, pred_abs_diff: {pred_abs_diff}")
            
            return loss, pred_abs_diff.mean(), pred_counts
        else:
            logging.info("Bernoulli loss enabled")

            audio_logits = torch.sigmoid(audio_logits)
            raw_pred_counts = torch.sum(audio_logits, dim=1)
            pred_counts = torch.round(raw_pred_counts)

            raw_pred_abs_diff = torch.abs(raw_pred_counts - gt_counts)
            pred_abs_diff = torch.abs(pred_counts - gt_counts)

            if PartialState().is_main_process:
                logging.info(f"gt_counts: {gt_counts}, raw_pred_counts: {raw_pred_counts}, raw_pred_abs_diff: {raw_pred_abs_diff}, pred_counts: {pred_counts}, pred_abs_diff: {pred_abs_diff}")

            loss = raw_pred_abs_diff.mean()

            return loss, pred_abs_diff.mean(), pred_counts

	# [min, max)
    def skip_example(self, example: Dict[str, Any], adapter: BaseDatasetAdapter) -> bool:
        if self.min_time is not None and adapter.get_num_speakers(example) < self.min_time:
            return True
        if self.max_time is not None and adapter.get_num_speakers(example) >= self.max_time: 
            return True
        return False