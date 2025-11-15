from typing import Dict, Type

from .base_dataset_adapter import BaseDatasetAdapter
from .librispeech import LibriSpeechAdapter
from .audiotime import AudioTimeAdapter
from .audioset import AudioSetAdapter
from .synthconv import SynthConvAdapter
from .libricount import LibriCountAdapter
from .groove_midi import GrooveMIDIAdapter

_ADAPTERS: Dict[str, Type[BaseDatasetAdapter]] = {
    "librispeech": LibriSpeechAdapter,
    "audiotime": AudioTimeAdapter,
    "audioset": AudioSetAdapter,
    "synthconv": SynthConvAdapter,
    "libricount": LibriCountAdapter,
    "groove_midi": GrooveMIDIAdapter,
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
    if "audioset" in repo:
        return "audioset"
    if "taudio" in repo:
        return "synthconv"
    if "libricount" in repo:
        return "libricount"
    if "groove_midi" in repo:
        return "groove_midi"
    raise ValueError(f"Cannot infer adapter from repository: {repository}")

