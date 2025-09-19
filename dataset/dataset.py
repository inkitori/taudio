import torch
from datasets import Dataset
from typing import Any, Dict, Optional
import torch.distributed as dist
import logging

from dataset import create_adapter, infer_adapter_from_repository
from models.base_model_adapter import BaseModelAdapter
from tasks.base_task import BaseTask

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

    def transform_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single example from the dataset on-the-fly.
        """
        # Since the dataloader's batch size is 1, the input `batch` dictionary 
        # contains lists of length 1. We extract the single example.
        example = {key: value[0] for key, value in batch.items()}
        
        # Apply the main processing logic to the single example
        processed_example = task.build_labels(
            example=example,
            ds_adapter=ds_adapter,
            model_adapter=model_adapter,
            eval_mode=False,
        )

        # The transform function expects a batch as output, so we wrap 
        # each value of the processed example in a list.
        return {key: [value] for key, value in processed_example.items()}

    if sharded:
        base_ds = ds_adapter.load_split(split).filter(lambda x: not task.skip_example(x, ds_adapter))
        world_size = dist.get_world_size()

        num_samples = len(base_ds)
        remainder = num_samples % world_size
        if remainder != 0:
            base_ds = base_ds.select(range(num_samples - remainder))

        sharded_ds = base_ds.shard(num_shards=dist.get_world_size(), index=dist.get_rank())

        # Apply the transform on-the-fly
        sharded_ds = sharded_ds.with_transform(transform_fn)
        
        dist.barrier()

        return sharded_ds
    else:
        base_ds = ds_adapter.load_split(split).filter(lambda x: not task.skip_example(x, ds_adapter))
            
        # Apply the transform on-the-fly
        base_ds = base_ds.with_transform(transform_fn)
            
        return base_ds