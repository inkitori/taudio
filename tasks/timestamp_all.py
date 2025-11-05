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
from accelerate import PartialState

from .base_task import BaseTask
from utils.utils import clamp, better_round
from utils.poisson import poisson_loss, infer_timestamps, infer_count, poisson_count_loss

STOPS = set(stopwords.words("english"))

class AllTimestampsTask(BaseTask):
    def __init__(self, *, key: str = "start", max_time: Optional[float] = None, min_time: Optional[float] = None):
        super().__init__()
        self.key = key
        self.max_time = max_time
        self.min_time = min_time

    def _build_conversation_text(self, *, model_processor: Any, ds_adapter: BaseDatasetAdapter, sorted_events, eval_mode: bool) -> str:
        prompt = ds_adapter.get_timestamp_all_prompt()

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

        sorted_events_list = []
        for event in sorted_events:
            event_name = ds_adapter.event_name(event)
            event_seconds = ds_adapter.get_target_seconds(event, self.key)

            event_object = {'event': event_name, 'start': event_seconds}
            sorted_events_list.append(event_object)

        sorted_events_list = json.dumps(sorted_events_list)

        # Training supervision: include expected JSON as assistant text when not in eval
        if not eval_mode:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": sorted_events_list},
                    ],
                }
            )

        logging.info(conversation)

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
        events = list(ds_adapter.get_events(example))
        sorted_events = ds_adapter.get_events_sorted(example)

        # Build prompt text via chat template and prepare inputs using the model adapter's processor
        processor = model_adapter.processor
        prompt_text = self._build_conversation_text(
            model_processor=processor, ds_adapter=ds_adapter, sorted_events=sorted_events, eval_mode=eval_mode)

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
        audio_labels_size = int((input_ids == model_adapter.audio_id).sum().item())
        audio_labels = torch.zeros(audio_labels_size, device=input_ids.device)

        for event in events:
            event_t_sec = ds_adapter.get_target_seconds(event, self.key)
            event_idx = clamp(math.floor(
                event_t_sec * (model_adapter.seconds_to_embedding)), 0, audio_labels_size - 1)
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

    def evaluate_tokens(
        self,
        *,
        example: Dict[str, Any],
        ds_adapter: BaseDatasetAdapter,
        model: Any,
        event: Optional[Dict[str, Any]] = None,
        error_bound: float = 0.1,
    ) -> Dict[str, Any]:
        gt_count = ds_adapter.get_num_speakers(example)
        
        inputs = self.build_labels(
            example=example,
            ds_adapter=ds_adapter,
            model_adapter=model.model_adapter,
            eval_mode=True,
        )
        inputs = inputs.to(next(model.parameters()).device)

        generated_string = model.generate(**inputs)
        logging.info(generated_string)
        try:
            parsed_events = json.loads(generated_string)
            pred_count = len(parsed_events)
        except Exception:
            return {"token_correct": 0.0, "parsing_error": 1.0}

        # Metric increments
        abs_err = abs(pred_count - gt_count)
        
        logging.info(f"Absolute error: {abs_err}, Error bound: {error_bound}, Correct: {1.0 if pred_count == gt_count else 0.0}")

        metrics: Dict[str, float] = {
            "token_abs_error_sum": abs_err,
            "token_correct": 1.0 if pred_count == gt_count else 0.0,
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
        gt_counts = (audio_labels == 1).sum(dim=1)

        if use_poisson_loss:
            timestamp_loss = poisson_loss(audio_logits, audio_labels, audio_labels_frame_mask).mean()
            count_loss = poisson_count_loss(audio_logits, gt_counts, audio_labels_frame_mask).mean()
            
            inferred_counts = infer_count(audio_logits, audio_labels_frame_mask)

            loss = timestamp_loss + count_loss

            if PartialState().is_main_process:
                logging.info(f"All Predicted Counts: {inferred_counts}, Ground Truth Counts: {gt_counts}")
        else:
            # # DONT FORGET TO HANDLE THE PADDING HERE TOO
            # if class_weighting:
            #     num_ones = (labels == 1).sum()
            #     num_zeros = (labels == 0).sum()
            #     pos_weight = (
            #         num_zeros / num_ones) if num_ones > 0 else 1.0

            #     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # else:
            #     criterion = nn.BCEWithLogitsLoss()

            # loss = criterion(logits, labels)
            # pred_timestamp = torch.argmax(logits).item() 
            # pred_timestamp = pred_timestamp + 0.5 # because we floor timestamps to the frame, we want to have full coverage over the frame
            pass


        return loss, (inferred_counts - gt_counts).abs().mean(), inferred_counts

    def skip_example(self, example: Dict[str, Any], adapter: BaseDatasetAdapter) -> bool:
        return False