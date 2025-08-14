from typing import Any, Dict, Iterable, Optional
import random
import logging
import torch
from nltk.corpus import stopwords
import json
import torch.nn as nn

from dataset.base_dataset_adapter import BaseDatasetAdapter
from models.base_model_adapter import BaseModelAdapter

from .base import BaseTask
from utils.utils import clamp, better_round
from utils.poisson import poisson_loss, infer_timestamps


STOPS = set(stopwords.words("english"))


class SingleTimestampTask(BaseTask):
    def __init__(self, key: str = "start", max_time: Optional[float] = None, min_time: Optional[float] = None):
        super().__init__()
        self.key = key
        self.max_time = max_time
        self.min_time = min_time

    def _choose_event(self, *, events: Iterable[Dict[str, Any]], ds_adapter: BaseDatasetAdapter, apply_fallback: bool = True) -> Dict[str, Any]:
        seen_names = set()
        candidate_events: list[Dict[str, Any]] = []
        unknown = set(ds_adapter.unknown_events())

        for event in events:
            name = ds_adapter.event_name(event)
            if name in seen_names:
                continue

            # timing filter
            t_seconds = float(event.get(self.key, 0.0))
            if self.min_time is not None and t_seconds < self.min_time:
                seen_names.add(name)
                continue
            if self.max_time is not None and t_seconds > self.max_time:
                seen_names.add(name)
                continue

            # lexical filters
            if name in unknown or name in STOPS:
                seen_names.add(name)
                continue

            candidate_events.append(event)
            seen_names.add(name)

        if len(candidate_events) > 0:
            return random.choice(candidate_events)

        if apply_fallback:
            # fallback to first event if no candidates
            return next(iter(events))
        else:
            return None

    def _build_conversation_text(self, *, model_processor: Any, ds_adapter: BaseDatasetAdapter, event: Dict[str, Any], eval_mode: bool) -> str:
        repo = (ds_adapter.repository or "").lower()
        name = ds_adapter.event_name(event)
        # repository-specific phrasing
        if "librispeech" in repo:
            prompt = f"What is the first occurence of the word '{name}'?"
        elif "audiotime" in repo:
            prompt = f"What is the first occurence of '{name}'?"
        else:
            raise ValueError(f"Unknown repository: {repo}")

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
        audio = ds_adapter.get_audio(example)
        events = list(ds_adapter.get_events(example))

        if len(events) == 0:
            raise ValueError("No events found in example")

        # If an event is provided (e.g., during eval), use it to keep GT and inputs aligned
        if event is None:
            event = self._choose_event(
                events=events, ds_adapter=ds_adapter, apply_fallback=not eval_mode)

        # Logging for traceability
        name = ds_adapter.event_name(event)
        t_sec = ds_adapter.get_target_seconds(event, self.key)
        logging.info(f"Selected event: {name}, {t_sec}")

        # Build prompt text via chat template and prepare inputs using the model adapter's processor
        processor = model_adapter.processor
        prompt_text = self._build_conversation_text(
            model_processor=processor, ds_adapter=ds_adapter, event=event, eval_mode=eval_mode)

        audio_frames = audio["array"]
        assert int(audio["sampling_rate"]) == 16000

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

        event_idx = clamp(better_round(
            t_sec * (model_adapter.seconds_to_embedding)), 0, labels_size - 1)
        labels[event_idx] = 1.0

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
            model_adapter=model.adapter,
            eval_mode=True,
            event=event,
        )
        inputs = inputs.to(next(model.parameters()).device)

        processor = model.adapter.processor
        with torch.no_grad():
            tokens = model.generate(
                **inputs, eos_token_id=processor.tokenizer.eos_token_id
            )
        generated_tokens = tokens[0][inputs["input_ids"].shape[1]:-1]
        generated_string = processor.tokenizer.decode(generated_tokens)
        try:
            token_pred = json.loads(generated_string)[name]
        except Exception:
            return {"token_correct": 0.0}

        # Metric increments
        abs_err = abs(float(token_pred) - float(gt))
        metrics: Dict[str, float] = {
            "token_abs_error_sum": abs_err,
            "token_correct": 1.0 if abs_err <= float(error_bound) else 0.0,
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

        gt = ds_adapter.get_target_seconds(event, self.key)

        # Build evaluation inputs via build_labels to mirror evaluate.py behavior
        inputs = self.build_labels(
            example=example,
            ds_adapter=ds_adapter,
            model_adapter=model.adapter,
            eval_mode=True,
            event=event,
        )
        inputs = inputs.to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model.adapter(
                **inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states[model.audio_layer]

            audio_hidden_states = hidden_states[inputs["input_ids"]
                                                == model.adapter.audio_id]

            logits = model.linear(audio_hidden_states).squeeze()
            if model.poisson_loss:
                aux_pred_top_idx = infer_timestamps(
                    1, logits.cpu().float().numpy())
            else:
                _, aux_pred_top_idx = torch.max(logits, dim=0)
        aux_pred = float(aux_pred_top_idx) / model.adapter.seconds_to_embedding

        logging.info(f"Auxiliary prediction: {aux_pred}, GT: {gt}")

        # Metric increments
        abs_err = abs(float(aux_pred) - float(gt))
        metrics: Dict[str, float] = {
            "aux_abs_error_sum": abs_err,
            "aux_correct": 1.0 if abs_err <= float(error_bound) else 0.0,
        }
        return metrics

    def calculate_loss(self, logits, labels, use_poisson_loss: bool = False, class_weighting: bool = False) -> torch.Tensor:
        if use_poisson_loss:
            return poisson_loss(logits.unsqueeze(0), labels.unsqueeze(0), torch.ones_like(logits.unsqueeze(0)))
        else:
            if class_weighting:
                num_ones = (labels == 1).sum()
                num_zeros = (labels == 0).sum()
                pos_weight = (
                    num_zeros / num_ones) if num_ones > 0 else 1.0

                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = nn.BCEWithLogitsLoss()

            return criterion(logits, labels)
