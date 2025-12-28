from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional
import logging
import math
import json
from accelerate import PartialState
from scipy.optimize import linear_sum_assignment
import torch
import textwrap
import re

from dataset.base_dataset_adapter import BaseDatasetAdapter
from dataset.librispeech import LibriSpeechAdapter
from models.base_model_adapter import BaseModelAdapter
from utils.utils import clamp, round_timestamp, round_timestamp_python
from utils.poisson import poisson_loss, infer_timestamps, infer_count

import numpy as np

from .base_task import BaseTask

THRESHOLDS = [0.005, 0.010, 0.020, 0.040, 0.050, 0.080, 0.100, 0.200]

def compute_matching_metrics(
    gt_timestamps: list,
    pred_timestamps: list,
    audio_length: float,
    prefix: str = "",  # e.g., "token_" for backwards compatibility
) -> list:
    """
    Compute metrics for all matching strategies.
    
    Returns a list of dicts, one per GT timestamp, with metrics for:
    - greedy_pairing: Process GTs in order, match to closest remaining prediction
    - strongly_aligned: Positional matching (GT[i] vs Pred[i])
    - l1_optimal: Hungarian algorithm minimizing total L1 error
    - acc_optimal: Hungarian algorithm minimizing errors above each threshold
    """
    if len(gt_timestamps) == 0:
        raise ValueError("Length of gt_timestamps should not be 0")

    gt_timestamps = sorted(gt_timestamps)
    pred_timestamps = sorted(pred_timestamps)[:len(gt_timestamps)] # we are truncating pred timestamps in case it is longer
    
    n_gt = len(gt_timestamps)
    n_pred = len(pred_timestamps)
    fallback_error = audio_length / 2
    
    gt_array = np.array(gt_timestamps)
    pred_array = np.array(pred_timestamps) if n_pred > 0 else np.array([])
    
    # Precompute distance matrix
    if n_pred > 0:
        distance_matrix = np.abs(gt_array[:, np.newaxis] - pred_array[np.newaxis, :])
    else:
        distance_matrix = None
    
    # Compute optimal matchings
    l1_optimal_pairing = {}
    acc_optimal_pairings = {t: {} for t in THRESHOLDS}
    
    if distance_matrix is not None:
        # L1-optimal
        gt_indices, pred_indices = linear_sum_assignment(distance_matrix)
        l1_optimal_pairing = dict(zip(gt_indices, pred_indices))
        
        # Accuracy-optimal per threshold
        for t in THRESHOLDS:
            cost_matrix = (distance_matrix > t).astype(float)
            gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
            acc_optimal_pairings[t] = dict(zip(gt_indices, pred_indices))
    
    metrics_list = []
    pred_remaining = list(pred_timestamps)
    
    for gt_idx, gt_ts in enumerate(gt_timestamps):
        metrics = {}
        
        # Helper to build metric keys with optional prefix
        def key(strategy, metric):
            return f"{strategy}/{prefix}{metric}"
        
        # --- Greedy pairing ---
        if len(pred_remaining) > 0:
            diffs = [abs(p - gt_ts) for p in pred_remaining]
            closest_idx = int(np.argmin(diffs))
            abs_err = round_timestamp_python(diffs[closest_idx])
            pred_remaining.pop(closest_idx)
            
            metrics[key("greedy_pairing", "parsing_error")] = 0.0
            metrics[key("greedy_pairing", "abs_error_sum")] = abs_err
            for t in THRESHOLDS:
                ms = int(t * 1000)
                metrics[key("greedy_pairing", f"correct_{ms}ms")] = 1.0 if abs_err <= t else 0.0
        else:
            metrics[key("greedy_pairing", "parsing_error")] = 1.0
            metrics[key("greedy_pairing", "abs_error_sum")] = fallback_error
            for t in THRESHOLDS:
                ms = int(t * 1000)
                metrics[key("greedy_pairing", f"correct_{ms}ms")] = 0.0
        
        # --- Strongly aligned ---
        if gt_idx < n_pred:
            abs_err = round_timestamp_python(abs(pred_timestamps[gt_idx] - gt_ts))
            metrics[key("strongly_aligned", "parsing_error")] = 0.0
            metrics[key("strongly_aligned", "abs_error_sum")] = abs_err
            for t in THRESHOLDS:
                ms = int(t * 1000)
                metrics[key("strongly_aligned", f"correct_{ms}ms")] = 1.0 if abs_err <= t else 0.0
        else:
            metrics[key("strongly_aligned", "parsing_error")] = 1.0
            metrics[key("strongly_aligned", "abs_error_sum")] = fallback_error
            for t in THRESHOLDS:
                ms = int(t * 1000)
                metrics[key("strongly_aligned", f"correct_{ms}ms")] = 0.0
        
        # --- L1-optimal ---
        if gt_idx in l1_optimal_pairing:
            pred_idx = l1_optimal_pairing[gt_idx]
            abs_err = round_timestamp_python(abs(pred_timestamps[pred_idx] - gt_ts))
            metrics[key("l1_optimal", "abs_error_sum")] = abs_err
            for t in THRESHOLDS:
                ms = int(t * 1000)
                metrics[key("l1_optimal", f"correct_{ms}ms")] = 1.0 if abs_err <= t else 0.0
        else:
            metrics[key("l1_optimal", "abs_error_sum")] = fallback_error
            for t in THRESHOLDS:
                ms = int(t * 1000)
                metrics[key("l1_optimal", f"correct_{ms}ms")] = 0.0
        
        # --- Accuracy-optimal ---
        for t in THRESHOLDS:
            ms = int(t * 1000)
            if gt_idx in acc_optimal_pairings[t]:
                pred_idx = acc_optimal_pairings[t][gt_idx]
                abs_err = round_timestamp_python(abs(pred_timestamps[pred_idx] - gt_ts))
                metrics[key("acc_optimal", f"correct_{ms}ms")] = 1.0 if abs_err <= t else 0.0
            else:
                metrics[key("acc_optimal", f"correct_{ms}ms")] = 0.0
        
        metrics_list.append(metrics)
    
    return metrics_list

def colorize_tensor_log(audio_labels, audio_logits, precision=2):
    """
    Colors for terminal output:
    - Green: positions where audio_labels == 1
    - Yellow: positions where audio_logits is nonzero but audio_labels != 1
    - Default: everything else
    """
    # ANSI codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    labels = audio_labels.float().squeeze().cpu().numpy()
    logits = audio_logits.float().squeeze().cpu().numpy()
    
    label_indices = set(np.where(labels == 1)[0])
    
    # Format audio_logits
    logit_strs = []
    for i, val in enumerate(logits):
        s = f"{val:.{precision}f}"
        if i in label_indices:
            s = f"{GREEN}{s}{RESET}"
        elif val >= 0.01:
            s = f"{YELLOW}{s}{RESET}"
        logit_strs.append(s)
    
    return (
        f"audio_logits: [{', '.join(logit_strs)}]"
    )

# Usage
def _collate_inputs(
    all_inputs: List[Dict[str, torch.Tensor]],
    model_adapter: BaseModelAdapter,
    padding_side: str 
) -> Dict[str, torch.Tensor]:
    # shouldn't need to be pad_token_id based on looking at qwen source but including to be safe
    pad_token_id = model_adapter.processor.tokenizer.pad_token_id
    max_len = max(inp["input_ids"].shape[1] for inp in all_inputs)
    
    input_ids_list = []
    attention_mask_list = []
    
    for inp in all_inputs:
        seq_len = inp["input_ids"].shape[1]
        pad_len = max_len - seq_len
        
        # Create padding tensors
        pad_ids = torch.full((1, pad_len), pad_token_id, dtype=inp["input_ids"].dtype, device=inp["input_ids"].device)
        pad_mask = torch.zeros((1, pad_len), dtype=inp["attention_mask"].dtype, device=inp["input_ids"].device)

        if pad_len > 0:
            if padding_side == "left":
                input_ids_list.append(torch.cat([pad_ids, inp["input_ids"]], dim=1))
                attention_mask_list.append(torch.cat([pad_mask, inp["attention_mask"]], dim=1))
            else:
                input_ids_list.append(torch.cat([inp["input_ids"], pad_ids], dim=1))
                attention_mask_list.append(torch.cat([inp["attention_mask"], pad_mask], dim=1))
        else:
            input_ids_list.append(inp["input_ids"])
            attention_mask_list.append(inp["attention_mask"])
    
    return {
        "input_ids": torch.cat(input_ids_list, dim=0),
        "attention_mask": torch.cat(attention_mask_list, dim=0),
        "input_features": torch.cat([inp["input_features"] for inp in all_inputs], dim=0),
        "feature_attention_mask": torch.cat([inp["feature_attention_mask"] for inp in all_inputs], dim=0),
    }

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

            # logging.info(f"Expected JSON\n{expected_json}")

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
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

        audio_length = ds_adapter.get_audio_frames(example).size / model.model_adapter.sampling_rate
        
        generated_string = model.generate(**inputs)

        logging.info(generated_string)
        
        # Compute metrics per example
        return self._compute_metrics_refactored(generated_string, events, ds_adapter, audio_length)
        

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
        batched_inputs = _collate_inputs(all_inputs, model.model_adapter, padding_side='left')
        batched_inputs = {k: v.to(next(model.parameters()).device) for k, v in batched_inputs.items()}
        
        generated_strings = model.generate_batch(**batched_inputs, max_new_tokens=4096)

        logging.info(generated_strings[-1])
        
        # Compute metrics per example
        all_metrics = []
        for generated_string, events, audio_length in zip(generated_strings, all_events, all_audio_lengths):
            metrics = self._compute_metrics_refactored(generated_string, events, ds_adapter, audio_length)
            all_metrics.extend(metrics)
        
        return all_metrics

    # def _compute_metrics(
    #     self,
    #     generated_string: str,
    #     events: List[Dict[str, Any]],
    #     ds_adapter: LibriSpeechAdapter,
    #     audio_length
    # ) -> Dict[str, float]:
    #     bad_metric_greedy_pairing = {
    #         "greedy_pairing_parsing_error": 1.0,
    #         "greedy_pairing_token_abs_error_sum": audio_length / 2,
    #         "greedy_pairing_token_correct_5ms": 0,
    #         "greedy_pairing_token_correct_10ms": 0,
    #         "greedy_pairing_token_correct_20ms": 0,
    #         "greedy_pairing_token_correct_40ms": 0,
    #         "greedy_pairing_token_correct_50ms": 0,
    #         "greedy_pairing_token_correct_80ms": 0,
    #         "greedy_pairing_token_correct_100ms": 0,
    #         "greedy_pairing_token_correct_200ms": 0,
    #     }
    #     bad_metric_strongly_aligned = {
    #         "strongly_aligned_parsing_error": 1.0,
    #         "strongly_aligned_token_abs_error_sum": audio_length / 2,
    #         "strongly_aligned_token_correct_5ms": 0,
    #         "strongly_aligned_token_correct_10ms": 0,
    #         "strongly_aligned_token_correct_20ms": 0,
    #         "strongly_aligned_token_correct_40ms": 0,
    #         "strongly_aligned_token_correct_50ms": 0,
    #         "strongly_aligned_token_correct_80ms": 0,
    #         "strongly_aligned_token_correct_100ms": 0,
    #         "strongly_aligned_token_correct_200ms": 0,
    #     }

    #     gt_sorted = sorted(ds_adapter.get_target_seconds(ev, self.key) for ev in events)
        
    #     pred_starts = self._parse_prediction_list(generated_string)
    #     if pred_starts is None:
    #         logging.error("Couldn't parse output: \n" + generated_string)
    #         logging.info("Falling back to regex output: ")

    #         pattern = r'([\d]+\.[\d]+)'
    #         pred_starts = [float(x) for x in re.findall(pattern, str(generated_string))]

    #         logging.info(pred_starts)
        
    #     pred_sorted_greedy_pairing = sorted(round_timestamp_python(float(p)) for p in pred_starts)
    #     pred_sorted_strongly_aligned = sorted(round_timestamp_python(float(p)) for p in pred_starts) 

    #     metrics = []

    #     for idx, gt in enumerate(gt_sorted):
    #         combined_metric = {}

    #         if idx >= len(pred_sorted_strongly_aligned):
    #             logging.error("Predicted less than the number of actual events in " + generated_string)
    #             combined_metric = bad_metric_strongly_aligned
    #         else:
    #             abs_err = round_timestamp_python(abs(pred_sorted_strongly_aligned[idx] - gt_sorted[idx]))
    #             combined_metric = {
    #                 "strongly_aligned_parsing_error": 0.0,
    #                 "strongly_aligned_token_abs_error_sum": abs_err,
    #                 "strongly_aligned_token_correct_5ms": 1.0 if abs_err <= 0.005 else 0.0,
    #                 "strongly_aligned_token_correct_10ms": 1.0 if abs_err <= 0.010 else 0.0,
    #                 "strongly_aligned_token_correct_20ms": 1.0 if abs_err <= 0.020 else 0.0,
    #                 "strongly_aligned_token_correct_40ms": 1.0 if abs_err <= 0.040 else 0.0,
    #                 "strongly_aligned_token_correct_50ms": 1.0 if abs_err <= 0.050 else 0.0,
    #                 "strongly_aligned_token_correct_80ms": 1.0 if abs_err <= 0.080 else 0.0,
    #                 "strongly_aligned_token_correct_100ms": 1.0 if abs_err <= 0.100 else 0.0,
    #                 "strongly_aligned_token_correct_200ms": 1.0 if abs_err <= 0.200 else 0.0,
    #             }

    #         # this is for the greedy pairing
    #         min_len = min(len(pred_sorted_greedy_pairing), len(gt_sorted))
    #         abs_errors = [
    #             round_timestamp_python(abs(pred_sorted_greedy_pairing[i] - gt))
    #             for i in range(min_len)
    #         ]

    #         if len(abs_errors) == 0:
    #             combined_metric = combined_metric | bad_metric_greedy_pairing
    #             metrics.append(combined_metric)
    #             logging.error("Predicted less than the number of actual events in " + generated_string)
    #             continue

    #         abs_err = min(abs_errors)
    #         abs_err_idx = abs_errors.index(abs_err) 
    #         pred_sorted_greedy_pairing.pop(abs_err_idx) 

    #         greedy_pairing_metric = {
    #             "greedy_pairing_parsing_error": 0.0,
    #             "greedy_pairing_token_abs_error_sum": abs_err,
    #             "greedy_pairing_token_correct_5ms": 1.0 if abs_err <= 0.005 else 0.0,
    #             "greedy_pairing_token_correct_10ms": 1.0 if abs_err <= 0.010 else 0.0,
    #             "greedy_pairing_token_correct_20ms": 1.0 if abs_err <= 0.020 else 0.0,
    #             "greedy_pairing_token_correct_40ms": 1.0 if abs_err <= 0.040 else 0.0,
    #             "greedy_pairing_token_correct_50ms": 1.0 if abs_err <= 0.050 else 0.0,
    #             "greedy_pairing_token_correct_80ms": 1.0 if abs_err <= 0.080 else 0.0,
    #             "greedy_pairing_token_correct_100ms": 1.0 if abs_err <= 0.100 else 0.0,
    #             "greedy_pairing_token_correct_200ms": 1.0 if abs_err <= 0.200 else 0.0,
    #         }

    #         combined_metric = combined_metric | greedy_pairing_metric

    #         metrics.append(combined_metric)
        
    #     return metrics
    
    def _compute_metrics_refactored(
        self,
        generated_string: str,
        events: list,
        ds_adapter,  # LibriSpeechAdapter
        audio_length: float,
    ) -> list:
        """Refactored to use shared compute_matching_metrics."""
        
        # Parse predictions
        pred_starts = self._parse_prediction_list(generated_string)
        if pred_starts is None:
            logging.error("Couldn't parse output: \n" + generated_string)
            pattern = r'([\d]+\.[\d]+)'
            pred_starts = [float(x) for x in re.findall(pattern, str(generated_string))]
            logging.info(f"Fallback regex output: {pred_starts}")
        
        # Get ground truth
        gt_sorted = sorted(ds_adapter.get_target_seconds(ev, self.key) for ev in events)
        pred_sorted = sorted(round_timestamp_python(float(p)) for p in pred_starts)
        
        # Use shared function with "token_" prefix for backwards compatibility
        return compute_matching_metrics(gt_sorted, pred_sorted, audio_length, prefix="token_")


    # def evaluate_auxiliary_outputs(
    #     self,
    #     *,
    #     example: Dict[str, Any],
    #     ds_adapter: BaseDatasetAdapter,
    #     model: Any,
    #     error_bound: float = 0.1,
    # ) -> Optional[Dict[str, Any]]:
    #     ds_adapter = self._validate_adapter(ds_adapter)
    #     events = self._extract_events_and_transcript(example=example, ds_adapter=ds_adapter)

    #     # setting eval_mode to false so that calculate_loss knows how many events there are
    #     inputs = self.build_labels(
    #         example=example,
    #         ds_adapter=ds_adapter,
    #         model_adapter=model.model_adapter,
    #         eval_mode=False,
    #     )
    #     inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    #     # because we set eval_mode = false (which returns unbatched tensors) we have to rebatch them
    #     for key in inputs.keys():
    #         inputs[key] = inputs[key].unsqueeze(0)

    #     with torch.no_grad():
    #         outputs = model(**inputs, inference=False)
    #         preds_dict = outputs.auxiliary_prediction

    #     gt_starts = [ds_adapter.get_target_seconds(ev, self.key) for ev in events]

    #     predictions_per_method = {}
    #     predictions_per_method_immutable = {}

    #     for method_name, pred_list in preds_dict.items():
    #         predictions_per_method[method_name] = pred_list[0].detach().clone() # we only pass in a single example
    #         predictions_per_method_immutable[method_name] = pred_list[0].detach().clone() 

    #     thresholds = [0.005, 0.010, 0.020, 0.040, 0.050, 0.080, 0.100, 0.200]

    #     metrics_list = []

    #     # Precompute distance matrices once per method
    #     distance_matrices = {}
    #     for method_name, method_preds in predictions_per_method_immutable.items():
    #         distance_matrices[method_name] = torch.abs(
    #             torch.tensor(gt_starts).unsqueeze(1) - method_preds.cpu().unsqueeze(0)
    #         ).numpy()

    #     # L1-optimal matching (one per method)
    #     l1_optimal_pairings = {}
    #     for method_name, dist_matrix in distance_matrices.items():
    #         gt_indices, pred_indices = linear_sum_assignment(dist_matrix)
    #         l1_optimal_pairings[method_name] = {
    #             gt_idx: pred_idx for gt_idx, pred_idx in zip(gt_indices, pred_indices)
    #         }

    #     # Accuracy-optimal matching (one per method per threshold)
    #     acc_optimal_pairings = {}
    #     for method_name, dist_matrix in distance_matrices.items():
    #         for t in thresholds:
    #             cost_matrix = (dist_matrix > t).astype(float)
    #             gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
    #             acc_optimal_pairings[(method_name, t)] = {
    #                 gt_idx: pred_idx for gt_idx, pred_idx in zip(gt_indices, pred_indices)
    #             }

    #     for idx, gt_timestamp in enumerate(gt_starts):
    #         all_metrics = {}

    #         for method_name, method_preds in predictions_per_method.items():
    #             method_preds_immutable = predictions_per_method_immutable[method_name]

    #             # --- Greedy pairing (existing) ---
    #             diffs = torch.abs(method_preds - gt_timestamp)
    #             closest_idx = torch.argmin(diffs).item()
    #             pred_timestamp = round_timestamp_python(method_preds[closest_idx].item())
    #             abs_err = round_timestamp_python(abs(pred_timestamp - gt_timestamp))

    #             method_preds[closest_idx] = torch.inf # so we don't select this prediction again

    #             all_metrics[f"greedy_pairing_{method_name}/aux_abs_error_sum"] = abs_err

    #             for t in thresholds:
    #                 ms_label = int(t * 1000)
    #                 all_metrics[f"greedy_pairing_{method_name}/aux_correct_{ms_label}ms"] = 1.0 if abs_err <= t else 0.0

    #             # --- Strongly aligned (existing) ---
    #             pred_immutable = round_timestamp_python(method_preds_immutable[idx].item())
    #             abs_err_immutable = round_timestamp_python(abs(pred_immutable - gt_timestamp))
    #             all_metrics[f"strongly_aligned_{method_name}/aux_abs_error_sum"] = abs_err_immutable
    #             for t in thresholds:
    #                 ms_label = int(t * 1000)
    #                 all_metrics[f"strongly_aligned_{method_name}/aux_correct_{ms_label}ms"] = 1.0 if abs_err_immutable <= t else 0.0

    #             # --- L1-optimal matching (new) ---
    #             l1_pred_idx = l1_optimal_pairings[method_name][idx]
    #             pred_l1 = round_timestamp_python(method_preds_immutable[l1_pred_idx].item())
    #             abs_err_l1 = round_timestamp_python(abs(pred_l1 - gt_timestamp))
    #             all_metrics[f"l1_optimal_{method_name}/aux_abs_error_sum"] = abs_err_l1
    #             for t in thresholds:
    #                 ms_label = int(t * 1000)
    #                 all_metrics[f"l1_optimal_{method_name}/aux_correct_{ms_label}ms"] = 1.0 if abs_err_l1 <= t else 0.0

    #             # --- Accuracy-optimal matching (new) ---
    #             for t in thresholds:
    #                 acc_pred_idx = acc_optimal_pairings[(method_name, t)][idx]
    #                 pred_acc = round_timestamp_python(method_preds_immutable[acc_pred_idx].item())
    #                 abs_err_acc = round_timestamp_python(abs(pred_acc - gt_timestamp))
    #                 ms_label = int(t * 1000)
    #                 all_metrics[f"acc_optimal_{method_name}/aux_correct_{ms_label}ms"] = 1.0 if abs_err_acc <= t else 0.0

    #         metrics_list.append(all_metrics)

    #     return metrics_list

    def evaluate_auxiliary_outputs(
        self,
        *,
        example: dict,
        ds_adapter,
        model,
        error_bound: float = 0.1,
    ):
        ds_adapter = self._validate_adapter(ds_adapter)
        events = self._extract_events_and_transcript(example=example, ds_adapter=ds_adapter)
        
        inputs = self.build_labels(
            example=example,
            ds_adapter=ds_adapter,
            model_adapter=model.model_adapter,
            eval_mode=False,
        )
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        for key in inputs:
            inputs[key] = inputs[key].unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(**inputs, inference=False, true_inference=True)
            preds_dict = outputs.auxiliary_prediction
        
        gt_starts = [ds_adapter.get_target_seconds(ev, self.key) for ev in events]
        audio_length = ds_adapter.get_audio_frames(example).size / model.model_adapter.sampling_rate
        
        # Initialize one dict per GT timestamp
        metrics_list = [{} for _ in gt_starts]
        
        for method_name, pred_list in preds_dict.items():
            pred_timestamps = [round_timestamp_python(p.item()) for p in pred_list[0].detach().cpu()]
            
            # Get metrics for this method
            method_metrics = compute_matching_metrics(
                gt_starts, 
                pred_timestamps, 
                audio_length,
            )
            
            # Merge into per-GT dicts with method-specific key names
            for gt_idx, m in enumerate(method_metrics):
                for k, v in m.items():
                    # k is like "greedy_pairing/abs_error_sum"
                    # Transform to "greedy_pairing/method_name/aux_abs_error_sum"
                    strategy, metric = k.split("/", 1)
                    new_key = f"{strategy}/{method_name}/aux_{metric}"
                    metrics_list[gt_idx][new_key] = v
        
        return metrics_list

    def calculate_loss(
        self,
        audio_logits,
        audio_labels,
        audio_labels_frame_mask,
        model_adapter: BaseModelAdapter,
        use_poisson_loss: bool,
        class_weighting: bool,
        true_inference=False
    ) -> torch.Tensor:
        # logging.info(colorize_tensor_log(audio_labels, torch.exp(audio_logits)))

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
            preds_dict_np = {'fake_prediction': np.array([0])}

            if true_inference:
                if use_poisson_loss:
                    preds_dict_np = infer_timestamps(n_pred, example_audio_logits.cpu().float().detach().numpy(), model_adapter.embedding_to_frame_adjusted_milliseconds)
                else:
                    raise ValueError("Not supporting BCE for all timestamps")

            # 2. Process all method predictions
            for method_name, pred_array in preds_dict_np.items():
                pred_array = torch.from_numpy(pred_array)

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
