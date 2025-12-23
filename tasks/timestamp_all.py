from typing import Any, Dict, Iterable, List, Optional
import logging
import math
import json
from accelerate import PartialState
import torch
import textwrap
import re

from dataset.base_dataset_adapter import BaseDatasetAdapter
from dataset.librispeech import LibriSpeechAdapter
from models.base_model_adapter import BaseModelAdapter
from utils.utils import clamp, round_timestamp, round_timestamp_python
from utils.poisson import poisson_loss, infer_timestamps, infer_count

from .base_task import BaseTask


class AllTimestampsTask(BaseTask):
    def __init__(self, *, key: str = "start"):
        super().__init__()
        self.key = key
        self.min_time = None
        self.max_time = None

    def _validate_adapter(self, ds_adapter: BaseDatasetAdapter) -> LibriSpeechAdapter:
        if not isinstance(ds_adapter, LibriSpeechAdapter):
            raise ValueError("AllTimestampsTask only supports the LibriSpeechAdapter.")
        return ds_adapter

    def _extract_events_and_transcript(
        self, *, example: Dict[str, Any], ds_adapter: LibriSpeechAdapter
    ) -> List[Dict[str, Any]]:
        events = ds_adapter.get_events_sorted(example)

        filtered: List[Dict[str, Any]] = []
        unknown = set(ds_adapter.unknown_events())
        for ev in events:
            name = ds_adapter.event_name(ev)
            # don't filter out <unk> so we don't end up with misaligned timestamps
            # if name in unknown:
            #     continue
            filtered.append(ev)

        if len(filtered) == 0:
            raise ValueError("No valid word events found for transcript.")

        return filtered

    def _build_conversation_text(
        self,
        *,
        model_processor: Any,
        transcript: str,
        expected_json: Optional[str],
        eval_mode: bool,
    ) -> str:
        user_prompt = f"Transcript:\n{transcript}\nBased on the transcript, output the timestamps for every word"
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
                    {"type": "text", "text": user_prompt},
                    {"type": "audio", "audio": "PLACEHOLDER AUDIO"},
                ],
            },
        ]

        if not eval_mode and expected_json is not None:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": expected_json},
                    ],
                }
            )

        return model_processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=eval_mode
        )

    def build_labels(
        self,
        *,
        example: Dict[str, Any],
        ds_adapter: BaseDatasetAdapter,
        model_adapter: BaseModelAdapter,
        eval_mode: bool,
    ) -> Dict[str, Any]:
        ds_adapter = self._validate_adapter(ds_adapter)

        audio_frames = ds_adapter.get_audio_frames(example)
        events = self._extract_events_and_transcript(example=example, ds_adapter=ds_adapter)

        words = [ds_adapter.event_name(ev) for ev in events]
        transcript = " ".join(words)

        # Training supervision JSON
        expected_json = None
        if not eval_mode:
            target = [
                {
                    ds_adapter.EVENT_NAME: ds_adapter.event_name(ev),
                    "start": ds_adapter.get_target_seconds(ev, self.key),
                }
                for ev in events
            ]
            expected_json = f"```json\n{json.dumps(target, indent=4)}\n```"

            logging.info(f"Expected JSON\n{expected_json}")

        processor = model_adapter.processor
        prompt_text = self._build_conversation_text(
            model_processor=processor,
            transcript=transcript,
            expected_json=expected_json,
            eval_mode=eval_mode,
        )

        # logging.info(f"[ALL] Prompt Text:\n{prompt_text}")

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

        num_unscaled_frames = int((input_ids == model_adapter.audio_id).sum().item())
        audio_labels_size = num_unscaled_frames * model_adapter.scaling_factor
        audio_labels = torch.zeros(audio_labels_size, device=input_ids.device)

        for ev in events:
            start_sec = ds_adapter.get_target_seconds(ev, self.key)
            frame_idx = clamp(
                math.floor(
                    start_sec
                    * (model_adapter.seconds_to_embedding * model_adapter.scaling_factor)
                ),
                0,
                audio_labels_size - 1,
            )
            audio_labels[frame_idx] = 1.0

        labels = input_ids.clone()
        assistant_idx = (input_ids == model_adapter.assistant_id).nonzero(as_tuple=True)[1][0]
        labels[0, : assistant_idx + 1] = -100

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "input_features": input_features[0],
            "feature_attention_mask": feature_attention_mask[0],
            "audio_labels": audio_labels,
            "labels": labels[0],
        }

    def _parse_prediction_list(self, generated: str) -> Optional[List[float]]:
        triple_backtick_match = re.search(r"```json(.*?)```", generated, re.DOTALL)
        json_candidate = triple_backtick_match.group(1).strip() if triple_backtick_match else None
        if json_candidate is None:
            brace_match = re.search(r"\[.*\]", generated, re.DOTALL)
            if brace_match:
                json_candidate = brace_match.group(0).strip()
        if json_candidate is None:
            return None
        try:
            parsed = json.loads(json_candidate)
            starts = []
            for item in parsed:
                start_val = item.get("start")
                if start_val is None:
                    continue
                starts.append(float(start_val))
            return starts if len(starts) > 0 else None
        except Exception:
            return None

    # ----- Evaluation helpers -----
    def evaluate_tokens(
        self,
        *,
        example: Dict[str, Any],
        ds_adapter: BaseDatasetAdapter,
        model: Any,
        error_bound: float = 0.1,
    ) -> Optional[Dict[str, Any]]:
        ds_adapter = self._validate_adapter(ds_adapter)
        events = self._extract_events_and_transcript(example=example, ds_adapter=ds_adapter)

        inputs = self.build_labels(
            example=example,
            ds_adapter=ds_adapter,
            model_adapter=model.model_adapter,
            eval_mode=True,
        )
        inputs = inputs.to(next(model.parameters()).device)

        generated_string = model.generate(**inputs)
        logging.info(f"[ALL] Token prediction: \n{generated_string}")

        pred_starts = self._parse_prediction_list(generated_string)
        if pred_starts is None:
            logging.info("[ALL] Failed to parse token prediction.")
            return {"parsing_error": 1.0}

        gt_starts = [ds_adapter.get_target_seconds(ev, self.key) for ev in events]
        gt_sorted = sorted(gt_starts)
        pred_sorted = sorted([round_timestamp_python(float(p)) for p in pred_starts])
        
        logging.info(f"[ALL] Ground Truth: {str(gt_sorted)}\n[ALL] Predictions: {str(pred_sorted)}")

        if len(pred_sorted) == 0 or len(gt_sorted) == 0:
            return {"parsing_error": 1.0}

        min_len = min(len(pred_sorted), len(gt_sorted))
        abs_errors = [
            round_timestamp_python(abs(pred_sorted[i] - gt_sorted[i])) for i in range(min_len)
        ]
        avg_err = sum(abs_errors) / float(min_len)

        metrics: Dict[str, float] = {
            "token_abs_error_sum": avg_err,
            "token_correct_50ms": 1.0
            if all(err <= 0.050 for err in abs_errors)
            else 0.0,
            "token_length_mismatch": 1.0 if len(pred_sorted) != len(gt_sorted) else 0.0,
            "parsing_error": 0.0
        }
        return metrics

    def evaluate_tokens_batched(
        self,
        *,
        examples: List[Dict[str, Any]],
        ds_adapter: BaseDatasetAdapter,
        model: Any,
        error_bound: float = 0.1,
    ) -> List[Dict[str, Any]]:
        ds_adapter = self._validate_adapter(ds_adapter)
        
        all_inputs = []
        all_events = []
        
        for example in examples:
            events = self._extract_events_and_transcript(example=example, ds_adapter=ds_adapter)
            all_events.append(events)
            
            inputs = self.build_labels(
                example=example,
                ds_adapter=ds_adapter,
                model_adapter=model.model_adapter,
                eval_mode=True,
            )
            all_inputs.append(inputs)
        
        # Batch the inputs
        batched_inputs = self._collate_inputs(all_inputs, model.model_adapter)
        batched_inputs = {k: v.to(next(model.parameters()).device) for k, v in batched_inputs.items()}
        
        generated_strings = model.generate_batch(**batched_inputs, max_new_tokens=2048)

        logging.info(generated_strings[-1])
        
        # Compute metrics per example
        all_metrics = []
        for generated_string, events in zip(generated_strings, all_events):
            metrics = self._compute_metrics(generated_string, events, ds_adapter)
            all_metrics.append(metrics)
        
        return all_metrics


    def _collate_inputs(
        self,
        all_inputs: List[Dict[str, torch.Tensor]],
        model_adapter: BaseModelAdapter,
    ) -> Dict[str, torch.Tensor]:
        """Left-pad input_ids/attention_mask, concatenate the rest."""
        
        pad_token_id = model_adapter.processor.tokenizer.pad_token_id or 0
        max_len = max(inp["input_ids"].shape[1] for inp in all_inputs)
        
        input_ids_list = []
        attention_mask_list = []
        
        for inp in all_inputs:
            seq_len = inp["input_ids"].shape[1]
            pad_len = max_len - seq_len
            
            if pad_len > 0:
                input_ids_list.append(torch.cat([
                    torch.full((1, pad_len), pad_token_id, dtype=inp["input_ids"].dtype),
                    inp["input_ids"]
                ], dim=1))
                attention_mask_list.append(torch.cat([
                    torch.zeros((1, pad_len), dtype=inp["attention_mask"].dtype),
                    inp["attention_mask"]
                ], dim=1))
            else:
                input_ids_list.append(inp["input_ids"])
                attention_mask_list.append(inp["attention_mask"])
        
        return {
            "input_ids": torch.cat(input_ids_list, dim=0),
            "attention_mask": torch.cat(attention_mask_list, dim=0),
            "input_features": torch.cat([inp["input_features"] for inp in all_inputs], dim=0),
            "feature_attention_mask": torch.cat([inp["feature_attention_mask"] for inp in all_inputs], dim=0),
        }


    def _compute_metrics(
        self,
        generated_string: str,
        events: List[Dict[str, Any]],
        ds_adapter: LibriSpeechAdapter,
    ) -> Dict[str, float]:
        """Extract metrics for a single example."""
        
        pred_starts = self._parse_prediction_list(generated_string)
        if pred_starts is None:
            return {"parsing_error": 1.0}
        
        gt_sorted = sorted(ds_adapter.get_target_seconds(ev, self.key) for ev in events)
        pred_sorted = sorted(round_timestamp_python(float(p)) for p in pred_starts)
        
        if len(pred_sorted) == 0 or len(gt_sorted) == 0:
            return {"parsing_error": 1.0}
        
        min_len = min(len(pred_sorted), len(gt_sorted))
        abs_errors = [
            round_timestamp_python(abs(pred_sorted[i] - gt_sorted[i]))
            for i in range(min_len)
        ]
        
        return {
            "token_abs_error_sum": sum(abs_errors) / min_len,
            "token_correct_50ms": 1.0 if all(err <= 0.050 for err in abs_errors) else 0.0,
            "token_length_mismatch": 1.0 if len(pred_sorted) != len(gt_sorted) else 0.0,
            "parsing_error": 0.0,
        }

    def evaluate_auxiliary_outputs(
        self,
        *,
        example: Dict[str, Any],
        ds_adapter: BaseDatasetAdapter,
        model: Any,
        error_bound: float = 0.1,
    ) -> Optional[Dict[str, Any]]:
        ds_adapter = self._validate_adapter(ds_adapter)
        events = self._extract_events_and_transcript(example=example, ds_adapter=ds_adapter)

        # Build full labels so the surrogate head knows how many timestamps to predict.
        inputs = self.build_labels(
            example=example,
            ds_adapter=ds_adapter,
            model_adapter=model.model_adapter,
            eval_mode=False,
        )
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, inference=False)

        gt_starts = torch.tensor(
            [ds_adapter.get_target_seconds(ev, self.key) for ev in events],
            device=next(model.parameters()).device,
            dtype=outputs.surrogate_loss.dtype,
        )

        pred_tensor = outputs.auxiliary_prediction
        if pred_tensor.dim() == 1:
            pred_tensor = pred_tensor.unsqueeze(0)
        preds = pred_tensor[0]
        preds = preds[preds != -100]

        if preds.numel() == 0:
            return {"aux_parsing_error": 1.0}

        preds = round_timestamp(preds)
        gt_sorted, _ = torch.sort(gt_starts)
        pred_sorted, _ = torch.sort(preds)
        min_len = min(gt_sorted.numel(), pred_sorted.numel())
        if min_len == 0:
            return {"aux_parsing_error": 1.0}
        abs_err = torch.abs(pred_sorted[:min_len] - gt_sorted[:min_len]).mean()

        metrics: Dict[str, float] = {
            "aux_abs_error_sum": round_timestamp_python(abs_err.item()),
            "aux_correct_50ms": 1.0 if abs_err.item() <= 0.050 else 0.0,
            "aux_length_mismatch": 1.0
            if gt_sorted.numel() != pred_sorted.numel()
            else 0.0,
        }
        return metrics

    def calculate_loss(
        self,
        audio_logits,
        audio_labels,
        audio_labels_frame_mask,
        model_adapter: BaseModelAdapter,
        use_poisson_loss: bool,
        class_weighting: bool,
    ) -> torch.Tensor:
        batch_size = audio_logits.size(0)
        device = audio_logits.device
        dtype = audio_logits.dtype
        denom = model_adapter.seconds_to_embedding * model_adapter.scaling_factor

        predicted_seconds: List[torch.Tensor] = []
        gt_seconds: List[torch.Tensor] = []
        per_example_deviation: List[torch.Tensor] = []

        if use_poisson_loss:
            loss = poisson_loss(audio_logits, audio_labels, audio_labels_frame_mask).mean()
        else:
            losses = []
            for example in range(batch_size):
                example_logits = audio_logits[example]
                example_labels = audio_labels[example]
                if (example_labels == -100).any():
                    cutoff = (example_labels == -100).nonzero(as_tuple=True)[0][0].item()
                    example_logits = example_logits[:cutoff]
                    example_labels = example_labels[:cutoff]
                if class_weighting:
                    num_ones = (example_labels == 1).sum()
                    num_zeros = (example_labels == 0).sum()
                    pos_weight = num_zeros / torch.clamp(num_ones, min=1)
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(dtype))
                else:
                    criterion = torch.nn.BCEWithLogitsLoss()
                losses.append(criterion(example_logits, example_labels.to(example_logits.dtype)))
            loss = torch.stack(losses).mean()

        for example in range(batch_size):
            example_logits = audio_logits[example]
            example_labels = audio_labels[example]
            if (example_labels == -100).any():
                cutoff = (example_labels == -100).nonzero(as_tuple=True)[0][0].item()
                example_logits = example_logits[:cutoff]
                example_labels = example_labels[:cutoff]

            gt_indices = (example_labels == 1).nonzero(as_tuple=True)[0]
            gt_sec = round_timestamp(gt_indices.to(dtype) / denom)
            gt_seconds.append(gt_sec)

            n_pred = gt_indices.numel()
            if n_pred == 0:
                predicted_seconds.append(torch.tensor([], device=device, dtype=dtype))
                per_example_deviation.append(torch.tensor(0.0, device=device, dtype=dtype))
                continue

            if use_poisson_loss:
                timestamps_dict = infer_timestamps(
                    n_pred, example_logits.cpu().float().detach().numpy()
                )
                pred_idx = timestamps_dict['posterior_mode']
                pred_sec = round_timestamp(
                    torch.tensor(pred_idx, device=device) / denom
                )

                torch.set_printoptions(precision=2, sci_mode=False, linewidth=160, threshold=10000)
                pred_count = infer_count(example_logits.unsqueeze(0), torch.ones_like(example_logits.unsqueeze(0)))
                gt_count = example_labels.sum()

                if PartialState().is_main_process:
                    logging.info(example_labels)
                    logging.info(torch.exp(example_logits))

            else:
                topk = torch.topk(example_logits, k=n_pred).indices.to(dtype)
                pred_sec = round_timestamp((topk + 0.5) / denom)

            predicted_seconds.append(pred_sec)

            min_len = min(pred_sec.numel(), gt_sec.numel())
            if min_len > 0:
                per_example_deviation.append(
                    torch.abs(pred_sec[:min_len] - gt_sec[:min_len]).mean()
                )
            else:
                per_example_deviation.append(torch.tensor(0.0, device=device, dtype=dtype))

        max_len = max((p.numel() for p in predicted_seconds), default=0)
        if max_len == 0:
            auxiliary_prediction = torch.full((batch_size, 0), 0.0, device=device, dtype=dtype)
        else:
            auxiliary_prediction = torch.full(
                (batch_size, max_len), -100.0, device=device, dtype=dtype
            )
            for idx, preds in enumerate(predicted_seconds):
                if preds.numel() > 0:
                    auxiliary_prediction[idx, : preds.numel()] = preds

        if len(per_example_deviation) == 0:
            auxiliary_deviation = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            auxiliary_deviation = torch.stack(per_example_deviation).mean()

        # logging.info(f"[ALL] GT Seconds: {gt_seconds}")
        # logging.info(f"[ALL] Predicted Seconds: {predicted_seconds}")

        return loss, auxiliary_deviation, auxiliary_prediction

    def skip_example(self, example: Dict[str, Any], adapter: BaseDatasetAdapter) -> bool:
        adapter = self._validate_adapter(adapter)
        try:
            _ = self._extract_events_and_transcript(example=example, ds_adapter=adapter)
        except ValueError:
            return True
        return False


