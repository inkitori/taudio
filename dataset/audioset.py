from typing import Any, Dict, Iterable, List
from datasets import load_dataset
from datasets.features import Audio

from .base_dataset_adapter import BaseDatasetAdapter


class AudioSetAdapter(BaseDatasetAdapter):
    def load_streaming_split(self, split: str):
        # streaming causes some weird utf-8 encoding issues
        ds = load_dataset(self.repository, split=split, streaming=True, trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        return ds

    def get_audio(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return example["audio"]

    def get_events(self, example: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError

    def event_name(self, event: Dict[str, Any]) -> str:
        raise NotImplementedError

    def get_target_seconds(self, event: Dict[str, Any], key: str) -> float:
        raise NotImplementedError

    def get_num_speakers(self, example: Dict[str, Any]) -> int:
        return len(example["labels"])

    def unknown_events(self) -> List[str]:
        return []

    def get_timestamp_single_prompt(self, event_name: str) -> str:
        raise NotImplementedError

    def get_speaker_count_prompt(self) -> str:
        return "How many events are there in the audio?"