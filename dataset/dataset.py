import torch
from datasets import Dataset
from typing import Any, Dict, Optional
import torch.distributed as dist
import logging

from dataset import create_adapter, infer_adapter_from_repository
from models.base_model_adapter import BaseModelAdapter
from tasks.base_task import BaseTask

WRITER_BATCH_SIZE = 512

def get_ds(
    model_adapter: BaseModelAdapter,
    repository: str,
    split: str,
    task: BaseTask,
    take_first: Optional[int] = None,
    sharded: bool = True,
) -> Dataset:
    ds_adapter = create_adapter(infer_adapter_from_repository(
        repository), sampling_rate=model_adapter.sampling_rate, repository=repository, take_first=take_first)

    def preprocess_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        return task.build_labels(
            example=example,
            ds_adapter=ds_adapter,
            model_adapter=model_adapter,
            eval_mode=False,
        )

    if sharded:
        base_ds = ds_adapter.load_split(split).filter(lambda x: not task.skip_example(x, ds_adapter))
        world_size = dist.get_world_size()

        num_samples = len(base_ds)
        remainder = num_samples % world_size
        if remainder != 0:
            base_ds = base_ds.select(range(num_samples - remainder))


        sharded_ds = base_ds.shard(num_shards=dist.get_world_size(), index=dist.get_rank())

        sharded_ds = sharded_ds.map(preprocess_fn, remove_columns=base_ds.column_names, writer_batch_size=WRITER_BATCH_SIZE)

        sharded_ds = sharded_ds.with_format("torch")
        
        dist.barrier()

        return sharded_ds
    else:
        from accelerate.utils import broadcast_object_list
        from accelerate import PartialState

        if PartialState().is_main_process:
            base_ds = ds_adapter.load_split(split).filter(lambda x: not task.skip_example(x, ds_adapter))
            
            base_ds = base_ds.map(preprocess_fn, remove_columns=base_ds.column_names, writer_batch_size=WRITER_BATCH_SIZE)

            base_ds = base_ds.with_format("torch")
            base_ds = [base_ds]
        else:
            base_ds = [None]
        
        base_ds = broadcast_object_list(base_ds, from_process=0)
        base_ds = base_ds[0]
        
        return base_ds


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    batch_keys = batch[0].keys()
    collated = {}

    for key in batch_keys:
        items = [item[key] for item in batch]
        if key == 'input_ids' or key == 'attention_mask':
            collated[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=0, padding_side='right')
        elif key == 'labels':
            collated[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=-100, padding_side='right')
        elif key == 'label_ids':
            collated[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=-100, padding_side='right')
        else:
            collated[key] = torch.stack(items)

    return collated
