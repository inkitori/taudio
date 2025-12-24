from collections import defaultdict
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
        raise NotImplementedError()

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
        all_audio_lengths = []
        
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

            all_audio_lengths.append(ds_adapter.get_audio_frames(example).size / model.model_adapter.sampling_rate)
        
        # Batch the inputs
        batched_inputs = self._collate_inputs(all_inputs, model.model_adapter)
        batched_inputs = {k: v.to(next(model.parameters()).device) for k, v in batched_inputs.items()}
        
        generated_strings = model.generate_batch(**batched_inputs, max_new_tokens=4096)

        logging.info(generated_strings[-1])
        
        # Compute metrics per example
        all_metrics = []
        for generated_string, events, audio_length in zip(generated_strings, all_events, all_audio_lengths):
            metrics = self._compute_metrics(generated_string, events, ds_adapter, audio_length)
            all_metrics.extend(metrics)
        
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
        audio_length
    ) -> Dict[str, float]:
        """Extract metrics for a single example."""

        bad_metric = {
            "parsing_error": 1.0,
            "token_abs_error_sum": audio_length / 2,
            "token_correct_5ms": 0,
            "token_correct_10ms": 0,
            "token_correct_20ms": 0,
            "token_correct_40ms": 0,
            "token_correct_50ms": 0,
            "token_correct_80ms": 0,
            "token_correct_100ms": 0,
            "token_correct_200ms": 0,
        }

        gt_sorted = sorted(ds_adapter.get_target_seconds(ev, self.key) for ev in events)
        
        pred_starts = self._parse_prediction_list(generated_string)
        if pred_starts is None:
            logging.error("Couldn't parse output: \n" + generated_string)
            logging.info("Falling back to regex output: ")

            pattern = r'([\d]+\.[\d]+)'
            pred_starts = [float(x) for x in re.findall(pattern, str(generated_string))]

            logging.info(pred_starts)
        
        pred_sorted = sorted(round_timestamp_python(float(p)) for p in pred_starts)

        metrics = []

        for gt in gt_sorted:
            min_len = min(len(pred_sorted), len(gt_sorted))
            abs_errors = [
                round_timestamp_python(abs(pred_sorted[i] - gt))
                for i in range(min_len)
            ]

            if len(abs_errors) == 0:
                metrics.append(bad_metric)
                logging.error("Predicted less than the number of actual events in " + generated_string)
                continue


            abs_err = min(abs_errors)
            abs_err_idx = abs_errors.index(abs_err) 
            pred_sorted.pop(abs_err_idx) 

            metric = {
                "parsing_error": 0.0,
                "token_abs_error_sum": abs_err,
                "token_correct_5ms": 1.0 if abs_err <= 0.005 else 0.0,
                "token_correct_10ms": 1.0 if abs_err <= 0.010 else 0.0,
                "token_correct_20ms": 1.0 if abs_err <= 0.020 else 0.0,
                "token_correct_40ms": 1.0 if abs_err <= 0.040 else 0.0,
                "token_correct_50ms": 1.0 if abs_err <= 0.050 else 0.0,
                "token_correct_80ms": 1.0 if abs_err <= 0.080 else 0.0,
                "token_correct_100ms": 1.0 if abs_err <= 0.100 else 0.0,
                "token_correct_200ms": 1.0 if abs_err <= 0.200 else 0.0,
            }

            metrics.append(metric)
        
        return metrics

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

		# because we set eval_mode = false (which returns unbatched tensors) we have to rebatch them
        for key in inputs.keys():
            inputs[key] = inputs[key].unsqueeze(0)

        with torch.no_grad():
            outputs = model(**inputs, inference=False)

            preds_dict = outputs.auxiliary_prediction

        gt_starts = [ds_adapter.get_target_seconds(ev, self.key) for ev in events]

        predictions_per_method = {}
        for method_name, pred_list in preds_dict.items():
            if len(pred_list) == 0:
                raise ValueError(f"No auxiliary predictions for method: {method_name}")
            predictions_per_method[method_name] = pred_list[0]

        reference_preds = predictions_per_method.get("posterior_mode")
        pred_count = reference_preds.numel() if reference_preds is not None else 0

        logging.info(f"[ALL] GT Events: {len(gt_starts)}, Pred Events: {pred_count}")

        thresholds = [0.005, 0.010, 0.020, 0.040, 0.050, 0.080, 0.100, 0.200]

        metrics_list = []

        for gt_timestamp in gt_starts:
            all_metrics = {}

            for method_name, method_preds in predictions_per_method.items():
                if method_preds.numel() == 0:
                    raise ValueError(f"Empty prediction tensor for {method_name}")

                diffs = torch.abs(method_preds - gt_timestamp)
                closest_idx = torch.argmin(diffs).item()
                pred_timestamp = round_timestamp_python(method_preds[closest_idx].item())
                abs_err = round_timestamp_python(abs(pred_timestamp - gt_timestamp))

                method_preds[closest_idx] = torch.inf # so we don't select this prediction again

                all_metrics[f"{method_name}/aux_abs_error_sum"] = abs_err

                for t in thresholds:
                    ms_label = int(t * 1000)
                    metric_key = f"{method_name}/aux_correct_{ms_label}ms"
                    all_metrics[metric_key] = 1.0 if abs_err <= t else 0.0

            metrics_list.append(all_metrics)

        out = ""
        for idx in range(pred_count):
            METHOD = 'smooth_20ms_boxcar_iterative_resmoothing'
            pred_starts = predictions_per_method[METHOD]
            
            gt_timestamp = gt_starts[idx]
            pred_timestamp = round_timestamp_python(pred_starts[idx].item())
            abs_err = round_timestamp_python(abs(pred_timestamp - gt_timestamp))

            out += f"[{METHOD}] Pred: {pred_timestamp}, GT: {gt_timestamp}, Err: {abs_err}, Correct: {abs_err < 0.040}\n"
        
        logging.info(f"Rank {PartialState().process_index}\n{out}")

        return metrics_list


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
        
        # Dictionary to store a list of predictions for every method
        # e.g., predictions_accumulator["smooth_20ms_gauss"] = [val1, val2, ...]
        predictions_accumulator = defaultdict(lambda: [])

        # --- Loss Calculation Logic ---
        if use_poisson_loss:
            loss = poisson_loss(audio_logits, audio_labels, audio_labels_frame_mask).mean()
        else:
            raise ValueError("Not supporting BCE for all timestamps")

        # --- Prediction & Metrics Logic ---
        for example in range(batch_size):
            example_audio_logits = audio_logits[example]
            example_audio_labels = audio_labels[example].to(dtype)

            # Truncate based on mask
            if (example_audio_labels == -100).any():
                neg_100_idx = (example_audio_labels == -100).nonzero(as_tuple=True)[0][0].item()
                example_audio_logits = example_audio_logits[:neg_100_idx]
                example_audio_labels = example_audio_labels[:neg_100_idx]

            n_pred = (example_audio_labels == 1).sum().item()
            # 1. Get dictionary of predictions from the new infer_timestamps
            # returns {"posterior_mode": [...], "smooth_20ms_boxcar": [...], etc}
            if use_poisson_loss:
                preds_dict_np = infer_timestamps(n_pred, example_audio_logits.cpu().float().detach().numpy())
            else:
                raise ValueError("Not supporting BCE for all timestamps")

            # 2. Process all method predictions
            for method_name, pred_array in preds_dict_np.items():
                pred_array = torch.from_numpy(pred_array)

                logging.info(f"[ALL] pred_array dtype: {pred_array.dtype}")

                pred_array = pred_array / (model_adapter.seconds_to_embedding * model_adapter.scaling_factor)
                pred_array = round_timestamp(pred_array)

                predictions_accumulator[method_name].append(pred_array)

        predicted_timestamps = dict(predictions_accumulator)
        abs_error = 0

        return loss, abs_error, predicted_timestamps

    def skip_example(self, example: Dict[str, Any], adapter: BaseDatasetAdapter) -> bool:
        adapter = self._validate_adapter(adapter)
        try:
            _ = self._extract_events_and_transcript(example=example, ds_adapter=adapter)
        except ValueError:
            return True
        return False