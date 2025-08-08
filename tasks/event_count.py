from typing import Any, Dict, Iterable, Optional
import torch

from .base import BaseTask


class EventCountTask(BaseTask):
    def __init__(self):
        super().__init__(name="event_count", key=None)

    def filter_events(self, events: Iterable[Dict[str, Any]]):
        return events

    def select_event(self, events: Iterable[Dict[str, Any]]):
        # Not used for counting; return None
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
        # Placeholder: no auxiliary labels for counting in current pipeline
        label_ids = input_ids.clone()
        assistant_idx = (input_ids == assistant_id).nonzero(
            as_tuple=True)[1][0]
        label_ids[0, : assistant_idx + 1] = -100
        return {
            "labels": torch.tensor([]),
            "label_ids": label_ids[0],
        }

