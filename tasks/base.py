from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional


class BaseTask(ABC):
    """Abstraction for task-specific example construction and labeling."""

    name: str

    def __init__(self, name: str, key: Optional[str] = None, max_time: Optional[float] = None, min_time: Optional[float] = None):
        self.name = name
        self.key = key
        self.max_time = max_time
        self.min_time = min_time

    @abstractmethod
    def filter_events(self, events: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        """Filter events for suitability for this task."""

    @abstractmethod
    def select_event(self, events: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select a single event for single-target tasks. Return None if not applicable."""

    @abstractmethod
    def build_labels(self, *,
                     adapter,
                     model_processor: Any,
                     input_ids: Any,
                     assistant_id: int,
                     audio_id: int,
                     audio_seconds_to_embedding: float,
                     event: Dict[str, Any],
                     key: Optional[str]) -> Dict[str, Any]:
        """Produce labels and any auxiliary tensors specific to the task."""

