import numpy as np
from datasets import load_dataset
from datasets import IterableDataset

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

def get_audio_bounds(input_ids, begin_audio_id, end_audio_id):
    # for some reason it doesn't seem like the +1 and -1 are being applied?
    start_audio_index = (input_ids == begin_audio_id).nonzero(as_tuple=True)[-1][0] + 1
    end_audio_index = (input_ids == end_audio_id).nonzero(as_tuple=True)[-1][0] - 1

    return start_audio_index, end_audio_index

def get_dataset_length(repository: str, split: str):
    dataset = load_dataset(repository, split=split)
    dataset_length = len(dataset)
    
    return dataset_length

def patch_dataset_length(dataset, length):
    # crazy hack to add __len__ to IterableDataset
    # https://stackoverflow.com/a/1647794
    def make_method(inst, _cls, meth, lm):
        inst.__class__ = type(_cls.__name__, (_cls,), {meth: lm})

    make_method(dataset, IterableDataset, "__len__", lambda self: length)

def better_round(n):
    return int(n + 0.5)