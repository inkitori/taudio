from typing import Any, Dict, Iterable, Optional
import random
import logging
import torch
from nltk.corpus import stopwords
import json
import torch.nn as nn
import math
from dataset.base_dataset_adapter import BaseDatasetAdapter
from models.base_model_adapter import BaseModelAdapter

from .base_task import BaseTask
from utils.utils import clamp, better_round, round_timestamp, round_timestamp_python
from utils.poisson import poisson_loss, infer_timestamps

STOPS = set(stopwords.words("english"))

class SingleTimestampTask(BaseTask):
    def __init__(self, *, key: str = "start", max_time: Optional[float] = None, min_time: Optional[float] = None):
        super().__init__()
        self.key = key
        self.max_time = max_time
        self.min_time = min_time
        self.rounding_factor = None

    def _choose_event(self, *, events: Iterable[Dict[str, Any]], ds_adapter: BaseDatasetAdapter, apply_fallback: bool, return_all: bool = False) -> Dict[str, Any]:
        seen_names = set()
        candidate_events = []
        unknown = set(ds_adapter.unknown_events())

        logging.debug(f"Min time: {self.min_time}, Max time: {self.max_time}, Key: {self.key}, Apply fallback: {apply_fallback}")

        for event in events:
            name = ds_adapter.event_name(event)
            if name in seen_names:
                continue

            seen_names.add(name)

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

        logging.debug(f"Candidate events: {candidate_events}")
        if len(candidate_events) > 0:
            return random.choice(candidate_events)

        raise ValueError("No candidate events found")

    def _build_conversation_text(self, *, model_processor: Any, ds_adapter: BaseDatasetAdapter, event: Dict[str, Any], eval_mode: bool) -> str:
        name = ds_adapter.event_name(event)
        prompt = ds_adapter.get_timestamp_single_prompt(name, self.key)

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
            word_json = '{"%s": %s}' % (name, seconds)
            conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"{word_json}"},
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
                events=events, ds_adapter=ds_adapter, apply_fallback=not eval_mode)
        logging.info(f"Events: {events}")
        logging.info(f"Event: {event}")

        # Logging for traceability
        name = ds_adapter.event_name(event)
        t_sec = ds_adapter.get_target_seconds(event, self.key)
        logging.debug(f"Selected event: {name}, {t_sec}")

        # Build prompt text via chat template and prepare inputs using the model adapter's processor
        processor = model_adapter.processor
        prompt_text = self._build_conversation_text(
            model_processor=processor, ds_adapter=ds_adapter, event=event, eval_mode=eval_mode)

        logging.info(f"Prompt text: {prompt_text}")

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

        # Labels aligned to <AUDIO> embeddings (now 10 ms per frame: 4 frames per 40ms embedding)
        num_unscaled_frames = int((input_ids == model_adapter.audio_id).sum().item())
        audio_labels_size = num_unscaled_frames * model_adapter.scaling_factor  # 4 x 10ms frames per 40ms token
        audio_labels = torch.zeros(audio_labels_size, device=input_ids.device)

        # Convert timestamp to 10ms frame index (100 frames per second)
        event_idx = clamp(math.floor(
            t_sec * (model_adapter.seconds_to_embedding * model_adapter.scaling_factor)), 0, audio_labels_size - 1)
        audio_labels[event_idx] = 1.0

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
        event: Optional[Dict[str, Any]] = None,
        error_bound: float = 0.1,
    ) -> Dict[str, Any]:
        # Choose event if not provided so we can compute GT consistently
        if event is None:
            events = list(ds_adapter.get_events(example))
            event = self._choose_event(
                events=events, ds_adapter=ds_adapter, apply_fallback=False)
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
        logging.info(f"Token prediction: {generated_string}, GT: {gt}")
        try:
            token_pred = json.loads(generated_string)[name]
        except Exception:
            abs_err = ds_adapter.get_audio_frames(example).size / (2 * model.model_adapter.sampling_rate)
            return {"token_abs_error_sum": abs_err, "token_correct": 0.0, "parsing_error": 1.0} 

        # Metric increments
        token_pred = round_timestamp_python(float(token_pred), self.rounding_factor)
        abs_err = round_timestamp_python(abs(float(token_pred) - float(gt)), self.rounding_factor)
        logging.info(f"Absolute error: {abs_err}, Error bound: {error_bound}, Correct: {1.0 if abs_err <= float(error_bound) else 0.0}")

        logging.info(f"Audio frames half size: {ds_adapter.get_audio_frames(example).size / (2 * model.model_adapter.sampling_rate)}")

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
            event = self._choose_event(
                events=events, ds_adapter=ds_adapter, apply_fallback=False)
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
            pred_timestamp = round_timestamp_python(outputs.auxiliary_prediction.item(), self.rounding_factor)

        logging.info(f"Auxiliary prediction: {pred_timestamp}, GT: {gt_timestamp}")

        # Metric increments
        abs_err = round_timestamp_python(abs(pred_timestamp - gt_timestamp), self.rounding_factor)
        logging.info(f"Absolute error: {abs_err}, Error bound: {error_bound}, Correct: {1.0 if abs_err <= float(error_bound) else 0.0}")
        metrics: Dict[str, float] = {
            "aux_abs_error_sum": abs_err,
            "aux_correct_5ms": 1.0 if abs_err <= 0.005 else 0.0,
            "aux_correct_10ms": 1.0 if abs_err <= 0.010 else 0.0,
            "aux_correct_20ms": 1.0 if abs_err <= 0.020 else 0.0,
            "aux_correct_40ms": 1.0 if abs_err <= 0.040 else 0.0,
            "aux_correct_50ms": 1.0 if abs_err <= 0.050 else 0.0,
            "aux_correct_80ms": 1.0 if abs_err <= 0.080 else 0.0,
            "aux_correct_100ms": 1.0 if abs_err <= 0.100 else 0.0,
            "aux_correct_200ms": 1.0 if abs_err <= 0.200 else 0.0,
        }
        return metrics

    def calculate_loss(self, audio_logits, audio_labels, audio_labels_frame_mask, model_adapter: BaseModelAdapter, use_poisson_loss: bool, class_weighting: bool, ds_adapter: BaseDatasetAdapter, audio_logits_2 = None) -> torch.Tensor:
        batch_size = audio_logits.size(0)
        gt_timestamps = torch.argmax(audio_labels, dim=1)
        # Convert from 10ms frame index to seconds (100 frames per second)
        gt_timestamps = gt_timestamps / (model_adapter.seconds_to_embedding * model_adapter.scaling_factor)
        gt_timestamps = round_timestamp(gt_timestamps, self.rounding_factor)

        if use_poisson_loss and class_weighting:
            p_loss = poisson_loss(audio_logits, audio_labels, audio_labels_frame_mask).mean()

            # When both Poisson loss and class weighting are enabled, predict timestamp
            # as the average of Poisson-inferred and binary-argmax timestamps.
            predicted_timestamps = torch.zeros(batch_size, device=audio_logits.device)

            for example in range(batch_size):
                example_audio_logits = audio_logits[example]
                example_audio_labels = audio_labels[example]
                example_audio_labels = example_audio_labels.to(example_audio_logits.dtype)

                if (example_audio_labels == -100).any():
                    neg_100_idx = (example_audio_labels == -100).nonzero(as_tuple=True)[0][0].item()
                    example_audio_logits = example_audio_logits[:neg_100_idx]
                    example_audio_labels = example_audio_labels[:neg_100_idx]

                # Poisson timestamp (frame index)
                poisson_frame_idx = infer_timestamps(1, example_audio_logits.cpu().float().detach().numpy())[0]
                poisson_frame_idx = torch.tensor(poisson_frame_idx, device=audio_logits.device, dtype=audio_logits.dtype)
                poisson_seconds = poisson_frame_idx / (model_adapter.seconds_to_embedding * model_adapter.scaling_factor)
                poisson_seconds = round_timestamp(poisson_seconds, self.rounding_factor)

                # Binary timestamp (frame index) with bias subtraction
                biased_logits = example_audio_logits # - linear_bias
                binary_frame_idx = torch.argmax(biased_logits).to(audio_logits.dtype)
                binary_frame_idx = binary_frame_idx + 0.5  # cover full frame width
                binary_seconds = binary_frame_idx / (model_adapter.seconds_to_embedding * model_adapter.scaling_factor)
                binary_seconds = round_timestamp(binary_seconds, self.rounding_factor)

                # Average in seconds, then round to final grid
                avg_seconds = (poisson_seconds + binary_seconds) / 2
                predicted_timestamps[example] = round_timestamp(avg_seconds, self.rounding_factor)

            loss = p_loss

        elif use_poisson_loss:
            loss = poisson_loss(audio_logits, audio_labels, audio_labels_frame_mask).mean()

            predicted_timestamps = torch.zeros(batch_size, device=audio_logits.device)
            
            for example in range(batch_size):
                example_audio_logits = audio_logits[example]
                example_audio_labels = audio_labels[example]
                example_audio_labels = example_audio_labels.to(example_audio_logits.dtype)
                
                if (example_audio_labels == -100).any():
                    neg_100_idx = (example_audio_labels == -100).nonzero(as_tuple=True)[0][0].item()
                    example_audio_logits = example_audio_logits[:neg_100_idx]
                    example_audio_labels = example_audio_labels[:neg_100_idx]

                predicted_timestamps[example] = infer_timestamps(1, example_audio_logits.cpu().float().detach().numpy())[0]

                # Convert from 10ms frame index to seconds (100 frames per second)
                predicted_timestamps[example] = predicted_timestamps[example] / (model_adapter.seconds_to_embedding * model_adapter.scaling_factor)

                predicted_timestamps[example] = round_timestamp(predicted_timestamps[example], self.rounding_factor)
        else:
            predicted_timestamps = torch.zeros(batch_size, device=audio_logits.device)
            loss = torch.zeros(batch_size, device=audio_logits.device)

            for example in range(batch_size):
                example_audio_logits = audio_logits[example]
                example_audio_labels = audio_labels[example]
                example_audio_labels = example_audio_labels.to(example_audio_logits.dtype)

                if (example_audio_labels == -100).any():
                    neg_100_idx = (example_audio_labels == -100).nonzero(as_tuple=True)[0][0].item()
                    example_audio_logits = example_audio_logits[:neg_100_idx]
                    example_audio_labels = example_audio_labels[:neg_100_idx]

                predicted_timestamps[example] = torch.argmax(example_audio_logits)
                predicted_timestamps[example] = predicted_timestamps[example] + 0.5 # because we floor timestamps to the frame, we want to have full coverage over the frame
                # Convert from 10ms frame index to seconds (100 frames per second)
                predicted_timestamps[example] = predicted_timestamps[example] / (model_adapter.seconds_to_embedding * model_adapter.scaling_factor)
                predicted_timestamps[example] = round_timestamp(predicted_timestamps[example], self.rounding_factor)

                if class_weighting:
                    num_ones = (example_audio_labels == 1).sum()
                    num_zeros = (example_audio_labels == 0).sum()
                    pos_weight = num_zeros / num_ones
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(example_audio_logits.dtype))
                else:
                    criterion = nn.BCEWithLogitsLoss()

                loss[example] = criterion(example_audio_logits, example_audio_labels)

        loss = loss.mean()

        logging.info(f"Predicted timestamps: {predicted_timestamps}, Ground truth timestamps: {gt_timestamps}")
        
        return loss, torch.abs(predicted_timestamps - gt_timestamps).mean(), predicted_timestamps

    def skip_example(self, example: Dict[str, Any], adapter: BaseModelAdapter) -> bool:
        events = self._choose_event(events=list(adapter.get_events(example)), ds_adapter=adapter, apply_fallback=False, return_all=True)
        if len(events) == 0:
            return True
        return False