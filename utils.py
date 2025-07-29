import numpy as np
from datasets import load_dataset

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

# Pad 16khz raw audio frames, where the encoded audio is 40ms per embedding


def pad_audio(audio_frames, padding):
    pad_length = 16 * 40 * padding
    audio_frames = np.pad(audio_frames, (0, pad_length), mode='constant')

    return audio_frames


def get_audio_bounds(input_ids, begin_audio_id, end_audio_id):
    # for some reason it doesn't seem like the +1 and -1 are being applied?
    start_audio_index = (input_ids == begin_audio_id).nonzero(as_tuple=True)[-1][0] + 1
    end_audio_index = (input_ids == end_audio_id).nonzero(as_tuple=True)[-1][0] - 1

    return start_audio_index, end_audio_index

def get_dataset_length(repository: str, split: str):
    dataset = load_dataset(repository, split=split)
    dataset_length = len(dataset)
    
    return dataset_length
