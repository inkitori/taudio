from typing import Dict, Type

from .base_task import BaseTask
from .timestamp_single import SingleTimestampTask
from .speaker_count import SpeakerCountTask
from .timestamp_all import AllTimestampsTask


_TASKS: Dict[str, Type[BaseTask]] = {
    "SINGLE_WORD_TIMESTAMP": SingleTimestampTask,
    "ALL_TIMESTAMPS": AllTimestampsTask,
    "SPEAKER_COUNT": SpeakerCountTask,
}


def create_task(task_type: str, **kwargs) -> BaseTask:
    if task_type not in _TASKS:
        raise ValueError(f"Unknown task: {task_type}")
    return _TASKS[task_type](**kwargs)
