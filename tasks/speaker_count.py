from typing import Any, Dict, Iterable, Optional
import torch

from .base import BaseTask


class SpeakerCountingTask(BaseTask):
    def __init__(self):
        super().__init__(name="speaker_count", key=None)

    def filter_events(self, events: Iterable[Dict[str, Any]]):
        # Not used; counting uses entire example
        return []

    def select_event(self, events: Iterable[Dict[str, Any]]):
        # Not applicable for counting
        return None

    def build_labels(self, *,
                     adapter,
                     model_processor: Any,
                     input_ids: torch.Tensor,
                     assistant_id: int,
                     audio_id: int,
                     audio_seconds_to_embedding: float,
                     event: Dict[str, Any],
                     key: Optional[str]):
        # Labels are multi-hot with exactly N ones, where N is number of speakers
        labels_size = int((input_ids == audio_id).sum().item())
        labels = torch.zeros(labels_size)

        # Store count in the first position; dataset preprocessor will pass count via event["num_speakers"]
        num_speakers = int(event.get("num_speakers", 1))
        num_speakers = max(0, min(num_speakers, labels_size))
        if num_speakers > 0:
            labels[:num_speakers] = 1.0

        label_ids = input_ids.clone()
        assistant_idx = (input_ids == assistant_id).nonzero(
            as_tuple=True)[1][0]
        label_ids[0, : assistant_idx + 1] = -100

        return {
            "labels": labels,
            "label_ids": label_ids[0],
        }

    def build_prompt(self,
                     *,
                     model_processor: Any,
                     adapter,
                     example: Dict[str, Any],
                     event: Optional[Dict[str, Any]],
                     eval_mode: bool,
                     key: Optional[str] = None) -> str:
        # Uses dataset info to phrase the question; event unused
        prompt = "How many distinct speakers are there in this audio?"

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

        # No supervision text for counting
        return model_processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=eval_mode
        )
