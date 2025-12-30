from typing import Dict, Type

from .base_task import BaseTask
from .timestamp_all import AllTimestampsTask
from .timestamp_single_any import SingleTimestampAnyTask

_TASKS: Dict[str, Type[BaseTask]] = {
    "SINGLE_WORD_TIMESTAMP_ANY": SingleTimestampAnyTask,
    "ALL_TIMESTAMPS": AllTimestampsTask,
}


def create_task(task_type: str, **kwargs) -> BaseTask:
    if task_type not in _TASKS:
        raise ValueError(f"Unknown task: {task_type}")
    return _TASKS[task_type](**kwargs)
