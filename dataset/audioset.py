from typing import Any, Dict, Iterable, List
from datasets import load_dataset
from datasets.features import Audio
import logging
import numpy as np

from .base_dataset_adapter import BaseDatasetAdapter
from utils.utils import round_timestamp_python
from utils.utils import remove_indices

train_exclude_indices = [16103, 23870, 52776, 58610, 68716]
eval_exclude_indices = [5929]

def start_filter_fn(example):
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

        effective_split = split
        if split == 'dev':
            effective_split = 'train'

        ds = load_dataset(self.repository, split=effective_split)
        
        if self.repository == "enyoukai/AudioSet-Strong":
            if effective_split == "train":
                ds = remove_indices(ds, train_exclude_indices)
            elif effective_split == "eval":
                ds = remove_indices(ds, eval_exclude_indices)
        
        if split in ['train', 'dev']:
            ds = ds.train_test_split(test_size=0.05, seed=42)
            if split == 'train':
                ds = ds['train']
            else:
                ds = ds['test']

        ds = ds.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        if self.take_first:
            ds = ds.select(range(self.take_first))
        
        if self.key == "start":
            ds = ds.filter(start_filter_fn)
        return ds

    def get_audio_frames(self, example: Dict[str, Any]) -> Dict[str, Any]:
        audio = example["audio"]["array"]
        pad_samples = int(self.left_padding * self.sampling_rate)
        if pad_samples > 0:
            zeros = np.zeros(pad_samples, dtype=audio.dtype)
            audio = np.concatenate([zeros, audio], axis=0)
        return audio
    
    def get_audio(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return example["audio"]

    def get_events(self, example: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        # The source data uses key 'words', each item has 'word' and timing fields like 'start'/'end'
        if self.key == "end":
            return [event for event in example["events"] if event.get('end') != 10]
        else:
            return example['events']

    def event_name(self, event: Dict[str, Any]) -> str:
        # There are <unk> tokens which the generic pipeline can filter if desired
        return event.get("event_name", "")

    def get_target_seconds(self, event: Dict[str, Any], key: str) -> float:
        # key could be 'start' or 'end'
        return round_timestamp_python(float(event[key]) + self.left_padding)

    def get_num_speakers(self, example: Dict[str, Any]) -> int:
        return len(example['events'])

    def unknown_events(self) -> List[str]:
        return []

    def get_timestamp_single_prompt(self, event_name: str, key: str) -> str:
        return f"What is the first occurence of the event '{event_name}'?"

    def get_speaker_count_prompt(self) -> str:
        return "How many events are there in the audio?"

    def get_timestamp_all_prompt(self) -> str:
        return "How many events are there in the audio?"

    def get_timestamp_single_base_prompt(self, event_name: str) -> str:
        return f"State exactly the timestamp in seconds when the first occurence of the event '{event_name}' occurs. The format should be as follows: '2.435', with the seconds followed by a decimal point and the milliseconds."

    def get_timestamp_single_any_prompt(self, event_name: str, key: str, ordinal: int) -> str:
        suffix = 'st' if ordinal == 1 else 'nd' if ordinal == 2 else 'rd' if ordinal == 3 else 'th'
        if key == "start":
            return f"When does the {ordinal}{suffix} occurrence of the event '{event_name}' occur?"
        elif key == "end":
            return f"When does the {ordinal}{suffix} occurrence of the event '{event_name}' end?"
        else:
            raise ValueError(f"Invalid key: {key}")
