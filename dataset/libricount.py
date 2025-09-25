from typing import Any, Dict, Iterable, List
from datasets import load_dataset
from datasets.features import Audio

from utils.utils import reprocess_and_split_dataset

from .base_dataset_adapter import BaseDatasetAdapter

class LibriCountAdapter(BaseDatasetAdapter):
    def load_streaming_split(self, split: str):
        ds = load_dataset(self.repository, split=split, streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        if self.take_first:
            ds = ds.take(self.take_first)
        return ds

    def load_split(self, split: str):
        ds = load_dataset(self.repository)
        ds = reprocess_and_split_dataset(ds)
        ds = ds[split]
        
        ds = ds.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        if self.take_first:
            ds = ds.select(range(self.take_first))
        return ds

    def get_audio(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return example["audio"]

    def get_events(self, example: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError

    def event_name(self, event: Dict[str, Any]) -> str:
        # There are <unk> tokens which the generic pipeline can filter if desired
        raise NotImplementedError

    def get_target_seconds(self, event: Dict[str, Any], key: str) -> float:
        # key could be 'start' or 'end'
        raise NotImplementedError

    def get_num_speakers(self, example: Dict[str, Any]) -> int:
        return example["k"]

    def unknown_events(self) -> List[str]:
        return []

    def get_timestamp_single_prompt(self, event_name: str) -> str:
        raise NotImplementedError

    def get_speaker_count_prompt(self) -> str:
        return "How many speakers are there in the audio?"