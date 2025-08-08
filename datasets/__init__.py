from typing import Dict, Type

from .base_dataset_adapter import BaseDatasetAdapter
from .librispeech import LibriSpeechAdapter
from .audiotime import AudioTimeAdapter


_ADAPTERS: Dict[str, Type[BaseDatasetAdapter]] = {
    "librispeech": LibriSpeechAdapter,
    "audiotime": AudioTimeAdapter,
}


def create_adapter(name: str, **kwargs) -> BaseDatasetAdapter:
    key = name.lower()
    if key not in _ADAPTERS:
        raise ValueError(f"Unknown dataset adapter: {name}")
    return _ADAPTERS[key](**kwargs)


def infer_adapter_from_repository(repository: str) -> str:
    repo = repository.lower()
    if "librispeech" in repo:
        return "librispeech"
    if "audiotime" in repo:
        return "audiotime"
    raise ValueError(f"Cannot infer adapter from repository: {repository}")

