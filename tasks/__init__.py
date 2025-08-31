from typing import Dict, Type

from .base import BaseTask
from .timestamp_single import SingleTimestampTask
from .speaker_count import SpeakerCountTask


_TASKS: Dict[str, Type[BaseTask]] = {
    "SINGLE_WORD_TIMESTAMP": SingleTimestampTask,
    "SPEAKER_COUNT": SpeakerCountTask,
}


def create_task(task_type: str, **kwargs) -> BaseTask:
    if task_type not in _TASKS:
        raise ValueError(f"Unknown task: {task_type}")
    return _TASKS[task_type](**kwargs)
