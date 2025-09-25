from typing import Any, Dict, Iterable, List
from datasets import load_dataset
from datasets.features import Audio

from .base_dataset_adapter import BaseDatasetAdapter

class LibriSpeechAdapter(BaseDatasetAdapter):
    def load_streaming_split(self, split: str):
        ds = load_dataset(self.repository, split=split, streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        if self.take_first:
            ds = ds.take(self.take_first)
        return ds

    def load_split(self, split: str):
        ds = load_dataset(self.repository, split=split)
        ds = ds.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        if self.take_first:
            ds = ds.select(range(self.take_first))
        return ds

    def get_audio(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return example["audio"]

    def get_events(self, example: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        # The source data uses key 'words', each item has 'word' and timing fields like 'start'/'end'
        return example["words"]

    def event_name(self, event: Dict[str, Any]) -> str:
        # There are <unk> tokens which the generic pipeline can filter if desired
        return event.get("word", "")

    def get_target_seconds(self, event: Dict[str, Any], key: str) -> float:
        # key could be 'start' or 'end'
        return float(event[key])

    def get_num_speakers(self, example: Dict[str, Any]) -> int:
        return len(example['words'])

    def unknown_events(self) -> List[str]:
        return ["<unk>"]

    def get_timestamp_single_prompt(self, event_name: str) -> str:
        return f"What is the first occurence of the word '{event_name}'?"

    def get_speaker_count_prompt(self) -> str:
        return "How many words are spoken in the audio?"

    def get_timestamp_all_prompt(self) -> str:
        return "How many words are spoken in the audio?"