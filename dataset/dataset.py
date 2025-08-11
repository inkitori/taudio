import torch
from datasets import Dataset
import random
from typing import Any, Dict, Optional
from nltk.corpus import stopwords
import logging

from utils.utils import clamp, better_round

from dataset import create_adapter, infer_adapter_from_repository
from models.base_model_adapter import BaseModelAdapter
from tasks.types import TaskType
from tasks.timestamp_single import SingleTimestampTask
from tasks.speaker_count import SpeakerCountingTask

STOPS = set(stopwords.words('english'))


def get_ds(
    model_adapter: BaseModelAdapter,
    repository: str,
    split: str,
    task: TaskType,
    key: Optional[str] = None,
    max_time: Optional[float] = None,
    max_count: Optional[int] = None,
) -> Dataset:
    # Unified dataset construction that delegates prompt/label logic to task classes
    if task not in {TaskType.SINGLE_WORD_TIMESTAMP, TaskType.SPEAKER_COUNTING}:
        raise ValueError(f"Task type {task} not supported in dataset builder")

    ds_adapter = create_adapter(infer_adapter_from_repository(
        repository), repository=repository)

    # Instantiate task
    if task == TaskType.SINGLE_WORD_TIMESTAMP:
        task_impl = SingleTimestampTask(key=key or "start", max_time=max_time)
    else:  # TaskType.SPEAKER_COUNTING
        task_impl = SpeakerCountingTask()

    def preprocess_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        return task_impl.build_labels(
            example=example,
            ds_adapter=ds_adapter,
            model_adapter=model_adapter,
            eval_mode=False,
        )

    base_ds = ds_adapter.load_streaming_split(split)
    ds = base_ds.map(preprocess_fn, remove_columns=base_ds.column_names)
    return ds


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    batch_keys = batch[0].keys()
    collated = {}

    for key in batch_keys:
        items = [item[key] for item in batch]
        if key == 'input_ids' or key == 'attention_mask':
            collated[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=0, padding_side='left')
        elif key == 'labels':
            collated[key] = torch.cat(items, dim=0)
        elif key == 'label_ids':
            collated[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=-100, padding_side='right')
        else:
            collated[key] = torch.stack(items)

    return collated
