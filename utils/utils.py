import numpy as np
from datasets import load_dataset
from datasets import IterableDataset
import logging
import torch

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

def get_audio_bounds(input_ids, begin_audio_id, end_audio_id):
    # Supports both single-sequence (1D) and batched (2D) input_ids.
    # Returns start/end indices that exclude the special audio boundary tokens.
    if input_ids.dim() == 1:
        start_audio_index = (input_ids == begin_audio_id).nonzero(as_tuple=True)[0][0].item() + 1
        end_audio_index = (input_ids == end_audio_id).nonzero(as_tuple=True)[0][-1].item() - 1
        return start_audio_index, end_audio_index

    # Batched case: compute per-example bounds
    batch_size = input_ids.size(0)
    starts = []
    ends = []
    for b in range(batch_size):
        row = input_ids[b]
        row_begin = (row == begin_audio_id).nonzero(as_tuple=True)[0]
        row_end = (row == end_audio_id).nonzero(as_tuple=True)[0]
        # Use first begin and last end to cover the full audio span, excluding boundary tokens
        start_audio_index = row_begin[0].item() + 1
        end_audio_index = row_end[-1].item() - 1
        starts.append(start_audio_index)
        ends.append(end_audio_index)
    return starts, ends

def get_dataset_length(repository: str, split: str, task=None, ds_adapter=None):
    dataset = load_dataset(repository, split=split)
    dataset_length = len(dataset)

    if task is not None and ds_adapter is not None:
        for example in dataset:
            if task.skip_example(example, ds_adapter):
                dataset_length -= 1

    logging.info(f"Dataset length: {dataset_length}")
    
    return dataset_length

def patch_dataset_length(dataset, length):
    # crazy hack to add __len__ to IterableDataset
    # https://stackoverflow.com/a/1647794
    def make_method(inst, _cls, meth, lm):
        inst.__class__ = type(_cls.__name__, (_cls,), {meth: lm})

    make_method(dataset, IterableDataset, "__len__", lambda self: length)

def better_round(n):
    return int(n + 0.5)

def round_timestamp(n):
    return torch.round(n * 1000) / 1000

def round_timestamp_python(n):
    return round(n * 1000) / 1000

import math
from datasets import Dataset, DatasetDict, concatenate_datasets

def reprocess_and_split_dataset(
    dataset_dict: DatasetDict,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42
) -> DatasetDict:
    """
    Combines all splits of a DatasetDict, shuffles, and then splits the result
    into 'train', 'val', and 'test' splits.

    Args:
        dataset_dict (DatasetDict): The input dataset dictionary to process.
                                    It can have any number of existing splits.
        train_size (float): The proportion of the dataset to allocate to the train split.
        val_size (float): The proportion of the dataset to allocate to the validation split.
        test_size (float): The proportion of the dataset to allocate to the test split.
        seed (int): The random seed for shuffling to ensure reproducibility.

    Returns:
        DatasetDict: A new dataset dictionary with 'train', 'val', and 'test' splits.
    """
    # 1. Validate that the split proportions sum to 1.0
    if not math.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(
            f"The sum of train_size, val_size, and test_size must be 1.0, "
            f"but it is {train_size + val_size + test_size}"
        )

    # 2. Combine all existing splits into a single dataset
    combined_dataset = concatenate_datasets(list(dataset_dict.values()))

    # 3. Shuffle the combined dataset
    shuffled_dataset = combined_dataset.shuffle(seed=seed)

    # 4. Perform the first split to separate the training set
    train_val_test_split = shuffled_dataset.train_test_split(
        test_size=(val_size + test_size)
    )

    train_split = train_val_test_split['train']
    val_test_temp_split = train_val_test_split['test']

    # 5. Perform the second split to separate validation and test sets
    if val_size + test_size > 0:
        relative_test_size = test_size / (val_size + test_size)
        val_test_split = val_test_temp_split.train_test_split(
            test_size=relative_test_size
        )
        val_split = val_test_split['train']
        test_split = val_test_split['test']
    else:
        # Handle the case where val and test sizes are zero
        val_split = val_test_temp_split.select(range(0))
        test_split = val_test_temp_split.select(range(0))

    # 6. Create and return the final DatasetDict
    return DatasetDict({
        'train': train_split,
        'dev': val_split,
        'test': test_split
    })

def remove_indices(dataset, exclude_indices):
    logging.info(f"Size of dataset: {len(dataset)}")
    dataset = dataset.select([i for i in range(len(dataset)) if i not in exclude_indices])
    logging.info(f"Size of dataset after removing indices: {len(dataset)}")

    return dataset