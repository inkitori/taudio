import torch
from datasets import Dataset
from typing import Any, Dict, Optional
import torch.distributed as dist

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


    if dist.get_rank() == 0:
        base_ds = ds_adapter.load_split(split)
        ds = base_ds.filter(lambda x: not task.skip_example(x, ds_adapter))
        ds = ds.map(preprocess_fn, remove_columns=base_ds.column_names, writer_batch_size=WRITER_BATCH_SIZE)
        ds = ds.with_format("torch")
        processed_ds = [ds]
    else:
        processed_ds = [None]
    
    dist.broadcast_object_list(processed_ds, src=0)
    ds = processed_ds[0]
    
    dist.barrier()

    return ds


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
