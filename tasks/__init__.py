from typing import Dict, Type

from .base import BaseTask
from .timestamp_single import SingleTimestampTask
from .speaker_count import SpeakerCountingTask


_TASKS: Dict[str, Type[BaseTask]] = {
    "single_word_timestamp": SingleTimestampTask,
    "speaker_count": SpeakerCountingTask,
}


def create_task(name: str, **kwargs) -> BaseTask:
    key = name.lower()
    if key not in _TASKS:
        raise ValueError(f"Unknown task: {name}")
    return _TASKS[key](**kwargs)
