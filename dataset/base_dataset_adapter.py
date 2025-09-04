from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional


class BaseDatasetAdapter(ABC):
    """Abstract interface for dataset-specific adaptation.

    Responsibilities:
    - Load a streaming dataset for a repository and split
    - Extract audio arrays and sampling rates from raw example
    - Produce a list of candidate events for a given task
    - Convert an event to a label target (seconds or index)
    - Build a prompt suitable for the current model
    """

    repository: Optional[str]

    def __init__(self, sampling_rate: int, repository: Optional[str] = None, take_first: Optional[int] = None) -> None:
        self.repository = repository
        self.sampling_rate = sampling_rate
        self.take_first = take_first

    @abstractmethod
    def load_streaming_split(self, split: str):
        """Return an iterable/streaming dataset for this repository and split."""

    @abstractmethod
    def get_audio(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Return dict with keys: array (1D float32) and sampling_rate (int)."""

    @abstractmethod
    def get_events(self, example: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        """Yield event dicts with at least keys: 'word' (or name) and 'start'/'end' or target fields per task."""

    @abstractmethod
    def event_name(self, event: Dict[str, Any]) -> str:
        """Return display/name string for an event."""

    @abstractmethod
    def get_target_seconds(self, event: Dict[str, Any], key: str) -> float:
        """Return ground-truth target in seconds for a given event and key (e.g., 'start', 'end')."""

    @abstractmethod
    def get_num_speakers(self, example: Dict[str, Any]) -> int:
        """Return the number of distinct speakers for the given example."""

    @abstractmethod
    def unknown_events(self) -> List[str]:
        """Return list of strings that should be ignored as events."""
    
    @abstractmethod
    def get_timestamp_single_prompt(self, event_name: str) -> str:
        """Return the prompt for the timestamp single task."""

    @abstractmethod
    def get_speaker_count_prompt(self) -> str:
        """Return the prompt for the speaker count task."""