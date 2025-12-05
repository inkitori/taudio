"""Build a labeled AudioSet dataset from the balanced release plus strong labels.

This script:
- loads the base audio + video_id pairs from `agkphysics/AudioSet` (train/test)
- loads strong labels from `audioset_train_strong.tsv` and `audioset_eval_strong.tsv`
- maps label IDs to human-readable event names via `ontology.json`
- adds an `events` field (list of {start, end, event_name}) to each example
- optionally pushes the resulting DatasetDict to the Hugging Face Hub
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

import os

# os.environ["HF_ENDPOINT"] = "http://localhost:5564"
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Add project root to Python path for local imports if needed.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
SCRIPT_DIR = Path(__file__).parent
ONTOLOGY_PATH = SCRIPT_DIR / "ontology.json"
TRAIN_TSV_PATH = SCRIPT_DIR / "audioset_train_strong.tsv"
EVAL_TSV_PATH = SCRIPT_DIR / "audioset_eval_strong.tsv"

# Base dataset and default repo name for pushing (update to your username).
BASE_DATASET_REPO = "agkphysics/AudioSet"
DEFAULT_TARGET_REPO = "your-username/AudioSet-Strong-Balanced"


def load_label_lookup(ontology_path: Path) -> Dict[str, str]:
    """Return a mapping from label id to human-readable event name."""
    with ontology_path.open("r") as f:
        ontology = json.load(f)
    return {entry["id"]: entry["name"] for entry in ontology}


def load_events_from_tsv(tsv_path: Path, label_lookup: Dict[str, str]) -> Dict[str, List[Dict]]:
    """Load strong labels from a TSV into a dict keyed by segment_id/video_id."""
    df = pd.read_csv(tsv_path, sep="\t")
    events_by_id: Dict[str, List[Dict]] = {}
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Reading {tsv_path.name}"):
        segment_id = row.segment_id.split("_")[0]  # e.g., "<video_id>_<start_ms>"
        label_id = row.label
        events_by_id.setdefault(segment_id, []).append(
            {
                "start": float(row.start_time_seconds),
                "end": float(row.end_time_seconds),
                "event_name": label_lookup.get(label_id, label_id),
            }
        )

    return events_by_id


def annotate_split(split: str, events_lookup: Dict[str, List[Dict]]):
    """Load a split from the base dataset and add `events` when available."""
    ds = load_dataset(BASE_DATASET_REPO, split=split)

    def add_events(example):
        video_id = example.get("video_id") or example.get("segment_id")
        example["events"] = events_lookup.get(video_id, [])
        return example

    ds = ds.map(add_events)
    # Drop examples that have no matching strong labels
    ds = ds.filter(lambda ex: len(ex["events"]) > 0)
    return ds


def push_to_hub(dataset: DatasetDict, repo_id: str):
    """Push the dataset to the Hugging Face Hub."""
    api = HfApi()
    print(f"Pushing dataset to {repo_id} ...")
    dataset.push_to_hub(repo_id, max_shard_size="512MB")
    print("Push complete.")


def main():
    label_lookup = load_label_lookup(ONTOLOGY_PATH)
    print(f"Loaded {len(label_lookup)} label mappings from ontology.")

    train_events = load_events_from_tsv(TRAIN_TSV_PATH, label_lookup)
    eval_events = load_events_from_tsv(EVAL_TSV_PATH, label_lookup)
    print(f"Train segment_ids with events: {len(train_events)}")
    print(f"Eval segment_ids with events: {len(eval_events)}")

    train_split = annotate_split("train", train_events)
    test_split = annotate_split("test", eval_events)

    print(train_split)
    print(train_split[:10])
    print(test_split)

    dataset = DatasetDict({"train": train_split, "test": test_split})
    print(dataset)

    # Optional push to hub if env var is set or DEFAULT_TARGET_REPO is customized.
    target_repo = os.getenv("HF_TARGET_REPO", DEFAULT_TARGET_REPO)

    push_to_hub(dataset, "enyoukai/AudioSet-Strong-Balanced")

    # if "your-username" not in target_repo:
    #     push_to_hub(dataset, target_repo)
    # else:
    #     print(
    #         "Skipping push_to_hub. Set HF_TARGET_REPO env var or update "
    #         "DEFAULT_TARGET_REPO to push automatically."
    #     )


if __name__ == "__main__":
    main()

