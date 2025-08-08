from typing import Dict, Type

from .base import BaseTask
from .timestamp_single import SingleTimestampTask
from .timestamp_multi import MultiTimestampTask
from .event_count import EventCountTask


_TASKS: Dict[str, Type[BaseTask]] = {
    "single_word_timestamp": SingleTimestampTask,
    "multi_word_timestamp": MultiTimestampTask,
    "event_count": EventCountTask,
}


def create_task(name: str, **kwargs) -> BaseTask:
    key = name.lower()
    if key not in _TASKS:
        raise ValueError(f"Unknown task: {name}")
    return _TASKS[key](**kwargs)

