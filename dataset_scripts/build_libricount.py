import os
import json
import math
import random
from collections import defaultdict
from typing import List

import numpy as np
from scipy.io import wavfile
from datasets import load_dataset, Audio

# ---------------------------
# Configurable parameters
# ---------------------------
OUTPUT_DIR = "mixtures_start_random_test"
METADATA_JSON = os.path.join(OUTPUT_DIR, "mixtures_metadata.json")
SAMPLE_RATE = 16000
MIX_DURATION_SEC = 10.0
N_PER_K = 200
KS = list(range(1, 11))
SEED = 12345
NORMALIZE_HEADROOM = 0.95
# ---------------------------

def load_example_dataset(split: str = "test_clean", sample_rate: int = 16000):
    """
    Load gilkeyio/librispeech-alignments lazily from Hugging Face.
    Audio is decoded only when accessed.
    """
    ds = load_dataset("gilkeyio/librispeech-alignments", split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
    return ds

def _to_mono(arr: np.ndarray) -> np.ndarray:
    return arr if arr.ndim == 1 else np.mean(arr, axis=-1)

def _astype_float01(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / 32768.0
    return arr.astype(np.float32)

def _float_to_int16(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)

def prepare_candidate_index(dataset, min_duration_sec: float, sample_rate: int):
    """Return mapping speaker_id -> list of row indices with utterances >= min_duration_sec."""
    min_samples = int(math.ceil(min_duration_sec * sample_rate))
    speaker_map = defaultdict(list)
    for idx in range(len(dataset)):
        n_samples = dataset[idx]["audio"]["array"].shape[0]
        if n_samples >= min_samples:
            speaker_id = dataset[idx]["id"].split("-")[0]
            speaker_map[speaker_id].append(idx)
    return speaker_map

def extract_first_10s_clip(example, sample_rate: int, duration_sec: float = 10.0):
    """
    Take the first 10s segment (t=0.0 to 10.0).
    Return float32 audio and the time of the first word onset if available.
    """
    audio = example["audio"]["array"]
    clip_len = int(round(duration_sec * sample_rate))
    segment = audio[:clip_len]
    segment = _astype_float01(_to_mono(segment))

    # First word start time if available
    first_word_start = None
    if "words" in example and example["words"]:
        # filter words that are real (exclude <unk>)
        valid_words = [w for w in example["words"] if "start" in w and isinstance(w["start"], (int, float))]
        if valid_words:
            first_word_start = float(valid_words[0]["start"])

    return segment, first_word_start

def mix_clips(clips: List[np.ndarray], headroom: float = NORMALIZE_HEADROOM) -> np.ndarray:
    stacked = np.stack(clips, axis=0)
    mixture = stacked.sum(axis=0)
    max_abs = np.max(np.abs(mixture))
    if max_abs > headroom:
        mixture *= headroom / max_abs
    return np.clip(mixture, -1.0, 1.0).astype(np.float32)

def write_wav(path: str, sample_rate: int, array_int16: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wavfile.write(path, sample_rate, array_int16)

def create_mixtures(dataset,
                    output_dir: str = OUTPUT_DIR,
                    sample_rate: int = SAMPLE_RATE,
                    mix_duration_sec: float = MIX_DURATION_SEC,
                    n_per_k: int = N_PER_K,
                    ks: List[int] = KS,
                    seed: int = SEED):

    rng = random.Random(seed)
    print(f"Filtering utterances shorter than {mix_duration_sec:.1f}s...")
    speaker_map = prepare_candidate_index(dataset, min_duration_sec=mix_duration_sec, sample_rate=sample_rate)
    speaker_ids = list(speaker_map.keys())
    print(f"Found {len(speaker_ids)} eligible speakers.")

    metadata = []
    global_count = 0

    for k in ks:
        print(f"\nCreating {n_per_k} mixtures with k = {k} speakers each...")
        for idx in range(n_per_k):
            chosen_speakers = rng.sample(speaker_ids, k)
            clips, components = [], []

            for j, spk in enumerate(chosen_speakers):
                rec_idx = rng.choice(speaker_map[spk])
                rec = dataset[rec_idx]
                clip, first_word_start = extract_first_10s_clip(
                    rec, sample_rate, duration_sec=mix_duration_sec
                )

                if j == 0:
                    # First speaker always starts at time 0
                    offset_sec = 0.0
                else:
                    # Random offset between 0 and k seconds
                    offset_sec = rng.uniform(0.0, 4.0)

                offset_samples = int(round(offset_sec * sample_rate))
                if offset_samples > 0:
                    pad = np.zeros(offset_samples, dtype=np.float32)
                    clip = np.concatenate([pad, clip])
                    clip = clip[: int(mix_duration_sec * sample_rate)]  # trim back to 10s
                    if first_word_start is not None:
                        first_word_start = round(first_word_start + offset_sec, 3)

                clips.append(clip)
                components.append({
                    "speaker_id": spk,
                    "utterance_id": rec["id"],
                    "utterance_length_sec": len(rec["audio"]["array"]) / sample_rate,
                    "first_word_start_sec": first_word_start,
                    "offset_sec": offset_sec
                })
                
            mixture_float = mix_clips(clips, headroom=NORMALIZE_HEADROOM)
            mixture_int16 = _float_to_int16(mixture_float)

            global_count += 1
            fname = f"mixture_k{k}_{idx:04d}.wav"
            fpath = os.path.join(output_dir, fname)
            write_wav(fpath, sample_rate, mixture_int16)

            metadata.append({
                "mixture_filename": fname,
                "mixture_path": fpath,
                "k": k,
                "components": components
            })

            if (idx + 1) % 500 == 0:
                print(f"  {idx+1}/{n_per_k} done for k={k}")

    with open(METADATA_JSON, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nDone. Wrote {global_count} mixtures. Metadata in {METADATA_JSON}")

if __name__ == "__main__":
    # Example: If your dataset is a pandas DataFrame called `df` loaded from somewhere,
    # convert it to list-of-dicts: records = df.to_dict(orient="records")
    #
    # For demonstration here's a placeholder load function. Replace with your loader:

    # Replace this with actual dataset load:
    dataset = load_example_dataset()

    if len(dataset) == 0:
        print("Dataset is empty in this example. Replace load_example_dataset() with actual loader.")
    else:
        create_mixtures(dataset,
                        output_dir=OUTPUT_DIR,
                        sample_rate=SAMPLE_RATE,
                        mix_duration_sec=MIX_DURATION_SEC,
                        n_per_k=N_PER_K,
                        ks=KS,
                        seed=SEED)