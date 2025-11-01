from typing import Any, Dict, Iterable, List
from datasets import load_dataset
from datasets.features import Audio, ClassLabel
import numpy as np
import logging

from utils.utils import round_timestamp_python

from .base_dataset_adapter import BaseDatasetAdapter

class LibriCountAdapter(BaseDatasetAdapter):
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

    def get_audio_frames(self, example: Dict[str, Any]) -> Dict[str, Any]:
        audio = example["audio"]["array"]
        logging.info(f"Audio shape: {audio.shape}")
        pad_samples = int(self.left_padding * self.sampling_rate)
        if pad_samples > 0:
            zeros = np.zeros(pad_samples, dtype=audio.dtype)
            audio = np.concatenate([zeros, audio], axis=0)
        logging.info(f"Audio shape after padding: {audio.shape}")
        return audio

    def get_events(self, example: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        events = []
        for speaker_idx, component in enumerate(example['components']):
            events.append({
                'speaker': f"{speaker_idx + 1}",
                'start': component['first_word_start_sec'],
            })
        return events

    def event_name(self, event: Dict[str, Any]) -> str:
        # There are <unk> tokens which the generic pipeline can filter if desired
        return event['speaker']

    def get_target_seconds(self, event: Dict[str, Any], key: str) -> float:
        # key could be 'start' or 'end'
        if key == 'end':
            raise ValueError("End key not supported for LibriCount")

        fixed_start = round_timestamp_python(float(event['start']) + self.left_padding, self.timestamp_rounding_factor())
        return fixed_start

    def get_num_speakers(self, example: Dict[str, Any]) -> int:
        return example["k"]

    def unknown_events(self) -> List[str]:
        return []

    def get_timestamp_single_prompt(self, event_name: str, key: str) -> str:
        suffix = 'st' if event_name == "1" else 'nd' if event_name == "2" else 'rd' if event_name == "3" else 'th'

        if key == "start":
            return f"When does the {event_name}{suffix} speaker start speaking?"
        elif key == "end":
            raise ValueError("End key not supported for LibriCount")

    def get_speaker_count_prompt(self) -> str:
        return "How many speakers are there in the audio?"

    def timestamp_rounding_factor(self) -> int:
        return 1000