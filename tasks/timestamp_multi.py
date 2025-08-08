from typing import Any, Dict, Iterable, Optional, List
import torch

from .base import BaseTask
from utils.utils import clamp, better_round
from utils.qwen2_5_omni_constants import SECONDS_TO_EMBEDDING


class MultiTimestampTask(BaseTask):
    def __init__(self, key: str = "start", max_time: Optional[float] = None, min_time: Optional[float] = None):
        super().__init__(name="multi_word_timestamp",
                         key=key, max_time=max_time, min_time=min_time)

    def filter_events(self, events: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        for ev in events:
            t = float(ev[self.key])
            if (self.min_time is None or t >= self.min_time) and (self.max_time is None or t <= self.max_time):
                yield ev

    def select_event(self, events: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # Multi-event task: selection is not used; returning None is acceptable
        return None

    def build_labels(self, *,
                     adapter,
                     model_processor: Any,
                     input_ids: torch.Tensor,
                     assistant_id: int,
                     audio_id: int,
                     audio_seconds_to_embedding: float = SECONDS_TO_EMBEDDING,
                     event: Dict[str, Any],
                     key: Optional[str] = None) -> Dict[str, Any]:
        # For simplicity, currently same as single timestamp but could be extended to multi-hot
        labels_size = int((input_ids == audio_id).sum().item())
        labels = torch.zeros(labels_size)

        seconds = adapter.get_target_seconds(event, key or self.key)
        event_idx = clamp(better_round(
            seconds * audio_seconds_to_embedding), 0, labels_size - 1)
        labels[event_idx] = 1.0

        label_ids = input_ids.clone()
        assistant_idx = (input_ids == assistant_id).nonzero(
            as_tuple=True)[1][0]
        label_ids[0, : assistant_idx + 1] = -100

        return {
            "labels": labels,
            "label_ids": label_ids[0],
        }

