from typing import Any, Dict, Iterable, List
from datasets import load_dataset

from .base_dataset_adapter import BaseDatasetAdapter


class AudioTimeAdapter(BaseDatasetAdapter):
    def load_streaming_split(self, split: str):
        return load_dataset(self.repository, split=split, streaming=True)

    def get_audio(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return example["audio"]

    def get_events(self, example: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        # In AudioTime, we call these "words" but they are general events with a 'word' label and timing
        return example["words"]

    def event_name(self, event: Dict[str, Any]) -> str:
        return event.get("word", "")

    def get_target_seconds(self, event: Dict[str, Any], key: str) -> float:
        return float(event[key])

    def get_num_speakers(self, example: Dict[str, Any]) -> int:
        speakers = set()
        # Common structure: example may have 'speaker' at top-level or per event
        if 'speaker' in example:
            val = example['speaker']
            if isinstance(val, list):
                speakers.update(val)
            else:
                speakers.add(val)
        else:
            for ev in self.get_events(example):
                spk = ev.get("speaker", None)
                if spk is not None:
                    speakers.add(spk)
        return len(speakers) if len(speakers) > 0 else 1

    def unknown_events(self) -> List[str]:
        return []
