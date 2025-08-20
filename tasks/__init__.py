from typing import Dict, Type

from .base import BaseTask
from .timestamp_single import SingleTimestampTask
from .speaker_count import SpeakerCountTask
from .types import TaskType


_TASKS: Dict[TaskType, Type[BaseTask]] = {
    TaskType.SINGLE_WORD_TIMESTAMP: SingleTimestampTask,
    TaskType.SPEAKER_COUNTING: SpeakerCountTask,
}


def create_task(task_type: TaskType, **kwargs) -> BaseTask:
    if task_type not in _TASKS:
        raise ValueError(f"Unknown task: {task_type}")
    return _TASKS[task_type](**kwargs)
