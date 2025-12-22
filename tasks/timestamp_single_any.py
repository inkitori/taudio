import os
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, List, Tuple
import random
import logging
import torch
import json
import torch.nn as nn
import math
import re
from nltk.corpus import stopwords
import textwrap
from pathlib import Path
import hashlib
from dataset.base_dataset_adapter import BaseDatasetAdapter
from models.base_model_adapter import BaseModelAdapter
from dataset.audioset import AudioSetAdapter

from .base_task import BaseTask
from utils.utils import clamp, round_timestamp, round_timestamp_python
from utils.poisson import poisson_loss, infer_timestamps, infer_timestamps_binary

STOPS = set(stopwords.words("english"))

class SingleTimestampAnyTask(BaseTask):
    def __init__(self, *, key: str = "start", max_time: Optional[float] = None, min_time: Optional[float] = None):
        super().__init__()
        self.key = key
        self.max_time = max_time
        self.min_time = min_time

    def _choose_event(
        self,
        *,
        events: Iterable[Dict[str, Any]],
        ds_adapter: BaseDatasetAdapter,
        apply_fallback: bool,
        return_all: bool = False,
        example
    ) -> Dict[str, Any]:
        candidate_events: List[Dict[str, Any]] = []
        unknown = set(ds_adapter.unknown_events())

        logging.debug(f"[ANY] Min time: {self.min_time}, Max time: {self.max_time}, Key: {self.key}, Apply fallback: {apply_fallback}")

        filtered_events = []
        if isinstance(ds_adapter, AudioSetAdapter):
            filtered_events = [event for event in events if event['start'] > 0]
        else:
            filtered_events = [event for event in events]

        for event in filtered_events:
            name = ds_adapter.event_name(event)

            # timing filter
            t_seconds = float(ds_adapter.get_target_seconds(event, self.key))
            if self.min_time is not None and t_seconds < self.min_time:
                continue
            if self.max_time is not None and t_seconds >= self.max_time:
                continue

            # lexical filters
            if name in unknown or name in STOPS:
                continue

            candidate_events.append(event)

        if return_all:
            return candidate_events

        logging.debug(f"[ANY] Candidate events: {candidate_events}")
        
        if len(candidate_events) > 0:
            chosen_event = random.choice(candidate_events) 

        selected_idx = example.get('selected')
        if selected_idx is not None:
            logging.info(f"[_choose_event] Selected index already precomputed, using index {example['selected']}")

            if selected_idx == -1:
                raise ValueError("No candidate events found (precomputed index was -1)")

            chosen_event = events[example['selected']]

            logging.info(f"[_choose_event] Using event: {chosen_event}")

        if chosen_event is not None: 
            return chosen_event

        raise ValueError("No candidate events found")

    def _compute_ordinal(
        self,
        *,
        all_events: List[Dict[str, Any]],
        ds_adapter: BaseDatasetAdapter,
        selected_event: Dict[str, Any],
    ) -> int:
        """Compute 1-based ordinal of selected_event among occurrences of the same event name.
        Ties on timestamp are broken by the original order within all_events.
        """
        target_name = ds_adapter.event_name(selected_event)
        selected_t = float(ds_adapter.get_target_seconds(selected_event, self.key))

        same_name_positions: List[Tuple[int, Dict[str, Any], float]] = []
        for idx, ev in enumerate(all_events):
            if ds_adapter.event_name(ev) == target_name:
                t = float(ds_adapter.get_target_seconds(ev, self.key))
                same_name_positions.append((idx, ev, t))

        # Determine position by counting how many occurrences strictly earlier,
        # plus ties that appear earlier in original order.
        selected_idx_in_all = None
        for idx, ev, _ in same_name_positions:
            if ev is selected_event:
                selected_idx_in_all = idx
                break
        if selected_idx_in_all is None:
            # Fallback: rely on timestamp only if identity matching fails (shouldn't happen)
            selected_idx_in_all = -1

        ordinal_minus_one = 0
        for idx, _ev, t in same_name_positions:
            if t < selected_t:
                ordinal_minus_one += 1
            elif t == selected_t and idx < selected_idx_in_all:
                ordinal_minus_one += 1

        return int(ordinal_minus_one + 1)

    def _build_conversation_text(
        self,
        *,
        model_processor: Any,
        ds_adapter: BaseDatasetAdapter,
        event: Dict[str, Any],
        ordinal: int,
        eval_mode: bool,
    ) -> str:
        name = ds_adapter.event_name(event)
        prompt = ds_adapter.get_timestamp_single_any_prompt(name, self.key, ordinal)

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
            seconds = ds_adapter.get_target_seconds(event, self.key)
            expected_json = textwrap.dedent('''\
            ```json
            [
                {"%s": "%s", "start": %s}
            ]
            ```''') % (ds_adapter.EVENT_NAME, name, seconds)

            conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"{expected_json}"},
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
        event: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        audio_frames = ds_adapter.get_audio_frames(example)
        events = list(ds_adapter.get_events(example))

        if len(events) == 0:
            raise ValueError("No events found in example")

        # If an event is provided (e.g., during eval), use it to keep GT and inputs aligned
        if event is None:
            event = self._choose_event(
                events=events, ds_adapter=ds_adapter, apply_fallback=not eval_mode, example=example
            )

        # logging.info(f"[ANY] Events: {events}")
        # logging.info(f"[ANY] Event: {event}")

        # Logging for traceability
        name = ds_adapter.event_name(event)
        t_sec = ds_adapter.get_target_seconds(event, self.key)
        ordinal = self._compute_ordinal(all_events=events, ds_adapter=ds_adapter, selected_event=event)
        logging.info(f"[ANY] Selected event: {name}, {t_sec}, ordinal={ordinal}")

        # Build prompt text via chat template and prepare inputs using the model adapter's processor
        processor = model_adapter.processor
        prompt_text = self._build_conversation_text(
            model_processor=processor, ds_adapter=ds_adapter, event=event, ordinal=ordinal, eval_mode=eval_mode
        )

        # logging.info(f"[ANY] Prompt text:\n {prompt_text}")

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

        # Labels aligned to <AUDIO> embeddings (10 ms per frame: scaling factor config)
        num_unscaled_frames = int((input_ids == model_adapter.audio_id).sum().item())
        audio_labels_size = num_unscaled_frames * model_adapter.scaling_factor
        audio_labels = torch.zeros(audio_labels_size, device=input_ids.device)

        # Convert timestamp to 10ms frame index (100 frames per second)
        event_idx = clamp(
            math.floor(t_sec * (model_adapter.seconds_to_embedding * model_adapter.scaling_factor)),
            0,
            audio_labels_size - 1,
        )
        audio_labels[event_idx] = 1.0

        # Mask out everything up to and including the assistant token
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

    # ----- Evaluation helpers -----
    def evaluate_tokens(
        self,
        *,
        example: Dict[str, Any],
        ds_adapter: BaseDatasetAdapter,
        model: Any,
        event: Optional[Dict[str, Any]] = None,
        error_bound: float = 0.1,
    ) -> Dict[str, Any]:
        # Choose event if not provided so we can compute GT consistently
        if event is None:
            events = list(ds_adapter.get_events(example))
            event = self._choose_event(events=events, ds_adapter=ds_adapter, apply_fallback=False, example=example)
            if event is None:
                return None

        name = ds_adapter.event_name(event)
        gt = ds_adapter.get_target_seconds(event, self.key)

        # Build evaluation inputs via build_labels to mirror evaluate.py behavior
        inputs = self.build_labels(
            example=example,
            ds_adapter=ds_adapter,
            model_adapter=model.model_adapter,
            eval_mode=True,
            event=event,
        )
        inputs = inputs.to(next(model.parameters()).device)

        generated_string = model.generate(**inputs)
        logging.info(f"[ANY] Token prediction: \n{generated_string}\n, GT: {gt}")

        triple_backtick_match = re.search(r"```json(.*?)```", generated_string, re.DOTALL)
        json_candidate = triple_backtick_match.group(1).strip() if triple_backtick_match else None
        if json_candidate is None:
            brace_match = re.search(r"\[.*\]", generated_string, re.DOTALL)
            if brace_match:
                json_candidate = brace_match.group(0).strip()
        try:
            token_pred = json.loads(json_candidate)[0]['start']
        except Exception:
            match = re.findall(r"\d+(?:\.\d+)?", generated_string)
            if match:
                token_pred = float(match[0])
                logging.info(f"[ANY] Fallback numeric extraction: {str(token_pred)}, GT: {str(gt)}")
            else:
                abs_err = ds_adapter.get_audio_frames(example).size / (2 * model.model_adapter.sampling_rate)
                logging.info(f"[ANY] Parsing Error (check generated_string)")
                return {"token_abs_error_sum": abs_err, "token_correct": 0.0, "parsing_error": 1.0}

        # Metric increments
        token_pred = round_timestamp_python(float(token_pred))
        abs_err = round_timestamp_python(abs(float(token_pred) - float(gt)))
        logging.info(
            f"[ANY] Absolute error: {abs_err}, Error bound: {error_bound}, Correct: {1.0 if abs_err <= float(error_bound) else 0.0}"
        )

        metrics: Dict[str, float] = {
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
        # Choose event if not provided for consistent GT
        if event is None:
            events = list(ds_adapter.get_events(example))
            event = self._choose_event(events=events, ds_adapter=ds_adapter, apply_fallback=False, example=example)
            if event is None:
                return None

        gt_timestamp = ds_adapter.get_target_seconds(event, self.key)

        # Build evaluation inputs via build_labels to mirror evaluate.py behavior
        inputs = self.build_labels(
            example=example,
            ds_adapter=ds_adapter,
            model_adapter=model.model_adapter,
            eval_mode=True,
            event=event,
        )
        inputs = inputs.to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model(**inputs, inference=True)

            preds_dict = outputs.auxiliary_prediction
            if not isinstance(preds_dict, dict):
                preds_dict = {"default": preds_dict}

            
        thresholds = [0.005, 0.010, 0.020, 0.040, 0.050, 0.080, 0.100, 0.200]

        all_metrics = {}

        for method_name, pred_tensor in preds_dict.items():
            # Extract scalar and round
            pred_timestamp = round_timestamp_python(pred_tensor.item())
            abs_err = round_timestamp_python(abs(pred_timestamp - gt_timestamp))

            # logging.info(f"[{method_name.upper()}] Pred: {pred_timestamp}, GT: {gt_timestamp}, Err: {abs_err}")

            # Add general error metric
            all_metrics[f"{method_name}/aux_abs_error_sum"] = abs_err
            
            # Add accuracy metrics for each threshold
            for t in thresholds:
                # Format key as "method/aux_correct_5ms" etc.
                ms_label = int(t * 1000)
                metric_key = f"{method_name}/aux_correct_{ms_label}ms"
                all_metrics[metric_key] = 1.0 if abs_err <= t else 0.0

        return all_metrics

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
        
        # Pre-calculate scaling factor
        scaling = model_adapter.seconds_to_embedding * model_adapter.scaling_factor
        
        gt_timestamps = torch.argmax(audio_labels, dim=1)
        gt_timestamps = round_timestamp(gt_timestamps / scaling)

        # Dictionary to store a list of predictions for every method
        # e.g., predictions_accumulator["smooth_20ms_gauss"] = [val1, val2, ...]
        predictions_accumulator = defaultdict(lambda: torch.zeros(batch_size, device=device))

        # --- Loss Calculation Logic ---
        if use_poisson_loss:
            loss = poisson_loss(audio_logits, audio_labels, audio_labels_frame_mask).mean()
        else:
            # Standard BCE Path
            losses = torch.zeros(batch_size, device=device)
            for example in range(batch_size):
                example_audio_logits = audio_logits[example]
                example_audio_labels = audio_labels[example].to(example_audio_logits.dtype)
                
                # Mask handling
                if (example_audio_labels == -100).any():
                    mask = example_audio_labels != -100
                    example_audio_logits = example_audio_logits[mask]
                    example_audio_labels = example_audio_labels[mask]

                if class_weighting:
                    num_ones = (example_audio_labels == 1).sum()
                    num_zeros = (example_audio_labels == 0).sum()
                    pos_weight = num_zeros / torch.clamp(num_ones, min=1.0)
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                else:
                    criterion = nn.BCEWithLogitsLoss()
                
                losses[example] = criterion(example_audio_logits, example_audio_labels)
            loss = losses.mean()

        # --- Prediction & Metrics Logic ---
        for example in range(batch_size):
            example_audio_logits = audio_logits[example]
            example_audio_labels = audio_labels[example].to(dtype)

            # Truncate based on mask
            if (example_audio_labels == -100).any():
                neg_100_idx = (example_audio_labels == -100).nonzero(as_tuple=True)[0][0].item()
                example_audio_logits = example_audio_logits[:neg_100_idx]
                example_audio_labels = example_audio_labels[:neg_100_idx]

            # 1. Get dictionary of predictions from the new infer_timestamps
            # returns {"posterior_mode": [...], "smooth_20ms_boxcar": [...], etc}
            if use_poisson_loss:
                preds_dict_np = infer_timestamps(1, example_audio_logits.cpu().float().detach().numpy())
            else:
                preds_dict_np = infer_timestamps_binary(1, example_audio_logits.cpu().float().detach().numpy())

            # 2. Process all method predictions
            for method_name, pred_array in preds_dict_np.items():
                frame_idx = pred_array[0] # n_pred=1, so we take index 0
                predictions_accumulator[method_name][example] = frame_idx
                predictions_accumulator[method_name][example] = predictions_accumulator[method_name][example] / (model_adapter.seconds_to_embedding * model_adapter.scaling_factor)
                predictions_accumulator[method_name][example] = round_timestamp(predictions_accumulator[method_name][example])
                # predictions_accumulator[method_name][example] = seconds

        # --- Final Return Values ---
        # Convert defaultdict back to a regular dict for the return
        predicted_timestamps = dict(predictions_accumulator)
        
        # Calculate absolute error specifically using posterior_mode as requested
        base_preds = predicted_timestamps["default"]
        abs_error = torch.abs(base_preds - gt_timestamps).mean()

        logging.info(f"[ANY] Base Predicted timestamps: {base_preds}")
        logging.info(f"[ANY] Ground truth timestamps: {gt_timestamps}")

        return loss, abs_error, predicted_timestamps


    def skip_example(self, example: Dict[str, Any], adapter: BaseModelAdapter) -> bool:
        events = self._choose_event(events=list(adapter.get_events(example)), ds_adapter=adapter, apply_fallback=False, return_all=True, example=example)  # type: ignore
        if len(events) == 0:
            return True
        return False

    def evaluate_tokens_base(self, example: Dict[str, Any], ds_adapter: BaseDatasetAdapter, model_adapter: BaseModelAdapter) -> Dict[str, Any]:

        events = list(ds_adapter.get_events(example))
        event = self._choose_event(events=events, ds_adapter=ds_adapter, apply_fallback=False, example=example)

        name = ds_adapter.event_name(event)
        gt = ds_adapter.get_target_seconds(event, self.key)
        audio = ds_adapter.get_audio(example)

        # Compute ordinal of this event among same-name occurrences
        ordinal = self._compute_ordinal(all_events=events, ds_adapter=ds_adapter, selected_event=event)

        # Build a base prompt suitable for token-level decoding
        base_question = ds_adapter.get_timestamp_single_any_prompt(name, self.key, ordinal)
        prompt_text = f'{base_question} Respond in JSON format with the time in seconds.'
        generation_prefix = textwrap.dedent(f'''\
        ```json
        [
            {{\"{ds_adapter.EVENT_NAME}\": \"{name}\", "start": ''')

        logging.info(f"[ANY] Prompt with prefix: {prompt_text} | Prefix: {generation_prefix}")

        inputs = model_adapter.build_base_inputs(prompt_text, audio, generation_prefix=generation_prefix)
        inputs = inputs.to(torch.cuda.current_device())

  
        try:
            from models.audio_flamingo3_adapter import AudioFlamingo3Adapter

            if isinstance(model_adapter, AudioFlamingo3Adapter):
                inputs['input_features'] = inputs['input_features'].to(model_adapter.dtype) # audio flamingo for some reason emits float32 input_features
        except ModuleNotFoundError:
            logging.info(f"[ANY] Couldn't find AudioFlamingo3Adapter, not importing (likely too old of transformers version)")

        generated_string = model_adapter.generate(**inputs, max_new_tokens=10, decode_tokens=True)
        full_generation = f"{generation_prefix}{generated_string}"
        # logging.info(f"[ANY] Generated string: {generated_string}")
        logging.info(f"[ANY] Full generation: \n{full_generation}")

        token_pred: Optional[float] = None

        triple_backtick_match = re.search(r"```json(.*?)```", full_generation, re.DOTALL)
        json_candidate = triple_backtick_match.group(1).strip() if triple_backtick_match else None
        if json_candidate is None:
            brace_match = re.search(r"\[.*\]", full_generation, re.DOTALL)
            if brace_match:
                json_candidate = brace_match.group(0).strip()

        if json_candidate:
            try:
                parsed = json.loads(json_candidate)
                raw_start = parsed[0]['start']

                # Check if it is a string containing ':' (e.g., "00:00:02.420")
                if isinstance(raw_start, str) and ':' in raw_start:
                    parts = raw_start.split(':')
                    token_pred = 0.0
                    for part in parts:
                        # Accumulate seconds: hours -> mins -> secs
                        token_pred = token_pred * 60 + float(part)
                else:
                    # Handle standard floats or strings like "2.42"
                    token_pred = float(raw_start)
            except Exception as e:
                logging.debug(f"[ANY] JSON parsing failed: {e}")
                token_pred = None

        if token_pred is None:
            match = re.findall(r"\d+(?:\.\d+)?", generated_string)
            if match:
                token_pred = float(match[0])
                logging.info(f"[ANY] Fallback numeric extraction: {str(token_pred)}, GT: {str(gt)}")

        audio_duration = audio["array"].size / audio["sampling_rate"]

        logging.info(f"[ANY] Audio Duration: {audio_duration}")

        if token_pred is None:
            token_pred = float(audio_duration / 2)
            logging.info(f"[ANY] Falling back to half the audio frames size: {token_pred}")

        # Metric increments
        token_pred = round_timestamp_python(float(token_pred))
        token_pred = clamp(token_pred, 0, audio_duration)

        abs_err = round_timestamp_python(abs(float(token_pred) - float(gt)))


        logging.info(f"[ANY] Token Prediction: {str(token_pred)}, Ground Truth: {str(gt)}")

        metrics: Dict[str, float] = {
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
        return metrics

    def select_indices(self, ds, ds_adapter, split):
        ds = ds.map(
            _deterministic_map_fn,
            with_indices=True,
            fn_kwargs={
                "repository": ds_adapter.repository,
                "split": split,
                "ds_adapter": ds_adapter,
                "min_time": self.min_time,
                "max_time": self.max_time,
                "key": self.key
            }
        )

        local_rank = int(os.environ.get("LOCAL_RANK", 0)) 
        filename = f"{ds_adapter.repository.replace('/', '-')}_{split}_{self.min_time}_{self.max_time}_{self.key}_rank-{local_rank}_indices.txt"

        indices_dir = Path("indices")
        indices_path = indices_dir / filename
        indices_dir.mkdir(parents=True, exist_ok=True)

        selected_indices = ds['selected']
        
        with indices_path.open("w") as f:
            for idx_val in selected_indices:
                f.write(f"{idx_val}\n")
        
        logging.info(f"Saved {len(selected_indices)} indices to {filename}")
        
        return ds

def _deterministic_map_fn(example, idx, repository, split, ds_adapter, min_time, max_time, key):
    events = ds_adapter.get_events(example)
    candidate_indices = []
    unknown = set(ds_adapter.unknown_events())

    for i, event in enumerate(events):
        if isinstance(ds_adapter, AudioSetAdapter):
            if event['start'] <= 0:
                logging.info(f"[ANY] Skipping event {event}")
                continue

        name = ds_adapter.event_name(event)

        # 2. Timing filters
        t_seconds = float(ds_adapter.get_target_seconds(event, key))
        if min_time is not None and t_seconds < min_time:
            continue
        if max_time is not None and t_seconds >= max_time:
            continue

        # 3. Lexical filters
        if name in unknown or name in STOPS:
            continue

        candidate_indices.append(i)

    # 4. Deterministic Choice Logic
    if not candidate_indices:
        return {"selected": -1}

    # Hash based on repo, split, and row index
    hash_input = f"{repository}_{split}_{idx}".encode()
    hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16)
    
    # Pick one index from the valid candidates
    # This returns the index relative to the original 'components' list
    chosen_meta_idx = hash_value % len(candidate_indices)
    actual_component_idx = candidate_indices[chosen_meta_idx]
    
    return {"selected": actual_component_idx}