from typing import Any, Dict, Iterable, List
from datasets import load_dataset
from datasets.features import Audio
import logging

from .base_dataset_adapter import BaseDatasetAdapter
from utils.utils import round_timestamp_python
from utils.utils import remove_indices

train_exclude_indices = [16103, 23870, 52776, 58610, 68716]
eval_exclude_indices = [5929]

def filter_fn(example):
    for event in example['events']:
        if event['start'] == 0:
            return False
    return True

class AudioSetAdapter(BaseDatasetAdapter):
    def load_streaming_split(self, split: str):
        ds = load_dataset(self.repository, split=split, streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        if self.take_first:
            ds = ds.take(self.take_first)
        return ds

    def load_split(self, split: str):
        if split == "test":
            split = "eval"

        ds = load_dataset(self.repository, split=split)
        ds = ds.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        if self.take_first:
            ds = ds.select(range(self.take_first))

        if self.repository == "enyoukai/AudioSet-Strong":
            if split == "train":
                ds = remove_indices(ds, train_exclude_indices)
            elif split == "eval":
                ds = remove_indices(ds, eval_exclude_indices)
        
        logging.info(f"Size of dataset before filtering: {len(ds)}")
        ds = ds.filter(filter_fn)
        logging.info(f"Size of dataset after filtering: {len(ds)}")
        return ds

    def get_audio_frames(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return example["audio"]["array"]

    def get_events(self, example: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        # The source data uses key 'words', each item has 'word' and timing fields like 'start'/'end'
        return example["events"]

    def event_name(self, event: Dict[str, Any]) -> str:
        # There are <unk> tokens which the generic pipeline can filter if desired
        return event.get("event_name", "")

    def get_target_seconds(self, event: Dict[str, Any], key: str) -> float:
        # key could be 'start' or 'end'
        return round_timestamp_python(float(event[key]))

    def get_num_speakers(self, example: Dict[str, Any]) -> int:
        return len(example['events'])

    def unknown_events(self) -> List[str]:
        return []

    def get_timestamp_single_prompt(self, event_name: str) -> str:
        return f"What is the first occurence of the event '{event_name}'?"

    def get_speaker_count_prompt(self) -> str:
        return "How many events are there in the audio?"

    def get_timestamp_all_prompt(self) -> str:
        return "How many events are there in the audio?"