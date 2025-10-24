import torch
from datasets import Dataset
from typing import Any, Dict, Optional
import torch.distributed as dist
import logging
from typing import Any, Dict, List

from dataset import create_adapter, infer_adapter_from_repository
from models.base_model_adapter import BaseModelAdapter
from tasks.base_task import BaseTask

def get_ds(
    model_adapter: BaseModelAdapter,
    repository: str,
    split: str,
    task: BaseTask,
    take_first: Optional[int] = None,
    left_padding: int = 0,
) -> Dataset:
    def transform_fn(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Processes a batch of examples from the dataset on-the-fly.
        """
        # A list to hold the processed examples
        processed_examples = []

        # Get the number of examples in the batch.
        # We can check the length of any of the lists in the batch dict.
        batch_size = len(next(iter(batch.values())))

        # Iterate through each example in the batch
        for i in range(batch_size):
            # Create a dictionary for the current example
            example = {key: value[i] for key, value in batch.items()}
            
            # Apply the main processing logic to the single example
            processed_example = task.build_labels(
                example=example,
                ds_adapter=ds_adapter,
                model_adapter=model_adapter,
                eval_mode=False,
            )
            processed_examples.append(processed_example)

        # Now, convert the list of processed example dicts into a single dict 
        # of lists (the batch format).
        processed_batch = {
            key: [dic[key] for dic in processed_examples]
            for key in processed_examples[0]
        }

        return processed_batch

    ds_adapter = create_adapter(infer_adapter_from_repository(
        repository), sampling_rate=model_adapter.sampling_rate, repository=repository, take_first=take_first, left_padding=left_padding)

    base_ds = ds_adapter.load_split(split).filter(lambda x: not task.skip_example(x, ds_adapter))
        
    base_ds = base_ds.with_transform(transform_fn)
        
    return base_ds, ds_adapter

def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    batch_keys = batch[0].keys()
    collated = {}

    for key in batch_keys:
        items = [item[key] for item in batch]
        if key == 'input_ids' or key == 'attention_mask':
            collated[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=0, padding_side='left')
        elif key == 'labels':
            collated[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=-100, padding_side='left')
        elif key == 'audio_labels':
            collated[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=-100, padding_side='right')
        else:
            collated[key] = torch.stack(items)

    return collated