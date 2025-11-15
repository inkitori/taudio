from typing import Any, Dict, Iterable, List
from datasets import load_dataset
from datasets.features import Audio

from .base_dataset_adapter import BaseDatasetAdapter
from utils.utils import round_timestamp_python


class GrooveMIDIAdapter(BaseDatasetAdapter):
    def load_streaming_split(self, split: str):
        ds = load_dataset(self.repository, split=split, streaming=True)  # type: ignore
        ds = ds.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        if self.take_first:
            ds = ds.take(self.take_first)
        return ds

    def load_split(self, split: str):
        # HF repo provides 'train' and 'test' splits directly
        ds = load_dataset(self.repository, split=split)  # type: ignore
        ds = ds.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        if self.take_first:
            ds = ds.select(range(self.take_first))
        return ds

    def get_audio_frames(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return example["audio"]["array"]

    def get_audio(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return example["audio"]

    def get_events(self, example: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        # Each event has {'beat': int, 'timestamp': float}
        return example.get("events", [])

    def event_name(self, event: Dict[str, Any]) -> str:
        # Use a readable label so ordinal grouping by same-name works (e.g., "beat 1")
        return f"beat"

    def get_target_seconds(self, event: Dict[str, Any], key: str) -> float:
        # Only 'start' timestamps exist; round to milliseconds as floats
        if key not in ("start", "timestamp"):
            raise ValueError(f"Unsupported key for GrooveMIDIAdapter: {key}")
        return round_timestamp_python(float(event["timestamp"]))

    def get_num_speakers(self, example: Dict[str, Any]) -> int:
        # Not used for this dataset; return number of events
        return len(example.get("events", []))

    def unknown_events(self) -> List[str]:
        return []

    def get_timestamp_single_prompt(self, event_name: str, key: str) -> str:
        if key == "start":
            return f"What is the first occurrence of {event_name}?"
        elif key == "end":
            raise ValueError("End timestamps are not available in Groove MIDI.")
        else:
            return f"What is the first occurrence of {event_name}?"

    def get_timestamp_single_any_prompt(self, event_name: str, key: str, ordinal: int) -> str:
        suffix = "st" if ordinal == 1 else "nd" if ordinal == 2 else "rd" if ordinal == 3 else "th"
        if key == "start":
            return f"When does the {ordinal}{suffix} beat occur?"
        elif key == "end":
            raise ValueError("End timestamps are not available in Groove MIDI.")
        else:
            return f"When does the {ordinal}{suffix} beat occur?"

    def get_speaker_count_prompt(self) -> str:
        return "How many unique beats are in the audio?"

