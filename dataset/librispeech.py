from typing import Any, Dict, Iterable, List
from datasets import load_dataset

from .base_dataset_adapter import BaseDatasetAdapter


class LibriSpeechAdapter(BaseDatasetAdapter):
    def load_streaming_split(self, split: str):
        return load_dataset(self.repository, split=split, streaming=True)

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
        # LibriSpeech split is typically single-speaker, but some versions may include speaker ids
        speakers = set()
        for ev in self.get_events(example):
            spk = ev.get("speaker", None)
            if spk is not None:
                speakers.add(spk)
        # Fallback: if no speaker metadata, assume 1 speaker
        return len(speakers) if len(speakers) > 0 else 1

    def unknown_events(self) -> List[str]:
        return ["<unk>"]
