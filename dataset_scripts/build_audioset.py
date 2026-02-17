"""Build a labeled AudioSet dataset from the base release plus strong labels from custom CSVs.
Optimized for high-latency cluster storage using Parallel Mapping.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set

from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets, Audio
from huggingface_hub import HfApi
from tqdm import tqdm

# Enable HF Transfer for speed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Add project root to Python path for local imports if needed.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).parent
BASE_DATASET_REPO = "agkphysics/AudioSet"


def load_events_from_csv(csv_path: Path) -> Dict[str, List[Dict]]:
    """Load strong labels from a custom CSV into a dict keyed by 'id' (video_id)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Label CSV not found at: {csv_path}")

    print(f"Reading strong labels from {csv_path.name}...")
    df = pd.read_csv(csv_path)

    events_by_id: Dict[str, List[Dict]] = {}
    
    # Pre-calculate relative times
    df['rel_start'] = (df['label_start'] - df['start_time']).round(3)
    df['rel_end'] = (df['label_end'] - df['start_time']).round(3)

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Processing {csv_path.name}"):
        video_id = str(row.id)
        events_by_id.setdefault(video_id, []).append(
            {
                "start": float(row.rel_start),
                "end": float(row.rel_end),
                "event_name": row.label_name,
            }
        )

    return events_by_id


def annotate_split(
    split_name: str, 
    events_lookup: Dict[str, List[Dict]], 
    base_repo: str, 
    limit: Optional[int] = None
) -> Dataset:
    # Use fewer cores if on a login node, or more if on a compute node
    # 16-32 is usually the sweet spot for I/O bound tasks
    num_cores = 16 
    
    target_ids: Set[str] = set(events_lookup.keys())
    print(f"[{split_name}] Target strong labels: {len(target_ids)}")

    print(f"[{split_name}] Loading base dataset...")
    # NOTE: Using standard config 'balanced_and_unbalanced_segments'
    ds = load_dataset(
        base_repo, 
        "full", 
        split=split_name, 
        streaming=False
    )

    # 1. CRITICAL: Disable Decoding
    # This ensures we work with raw bytes (fast) instead of decoding audio (slow)
    print(f"[{split_name}] Disabling audio decoding for speed...")
    ds = ds.cast_column("audio", Audio(decode=False))

    # 2. Extract IDs for filtering (Read text column only)
    print(f"[{split_name}] extracting IDs for fast filtering...")
    id_col = "video_id" if "video_id" in ds.column_names else "segment_id"
    all_ids = ds[id_col] 
    
    # 3. Find Indices to Keep
    indices_to_keep = [
        i for i, vid in enumerate(tqdm(all_ids, desc=f"[{split_name}] Matching IDs"))
        if vid in target_ids
    ]
    print(f"[{split_name}] Found {len(indices_to_keep)} matches out of {len(all_ids)} rows.")

    # 4. Select the rows (Creates a View)
    ds = ds.select(indices_to_keep)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    # 5. Add events using PARALLEL MAP
    # We use map() instead of add_column() because add_column is single-threaded.
    # map() with num_proc allows us to fetch the scattered audio bytes in parallel.
    
    def add_events_batch(batch):
        ids = batch[id_col]
        # Look up events for the whole batch
        batch_events = [events_lookup.get(vid, []) for vid in ids]
        return {"events": batch_events}

    print(f"[{split_name}] Annotating and consolidating dataset ({num_cores} cores)...")
    ds = ds.map(
        add_events_batch,
        batched=True,        # Reduces Python overhead
        batch_size=1000,     # Process 1000 rows at a time
        num_proc=num_cores,  # PARALLEL I/O
        desc=f"[{split_name}] Processing"
    )

    # 6. Re-enable decoding metadata (Instant)
    # The final dataset will now automatically decode audio when accessed
    ds = ds.cast_column("audio", Audio(decode=True))
    
    return ds


def push_to_hub(dataset: DatasetDict, repo_id: str):
    print(f"\nPushing dataset to {repo_id} ...")
    dataset.push_to_hub(repo_id, max_shard_size="512MB")
    print("Push complete.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load Labels
    try:
        train_events = load_events_from_csv(args.train_csv)
        test_events = load_events_from_csv(args.test_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 2. Process Splits
    
    # Uncomment bal_train when ready
    bal_train_split = annotate_split("bal_train", train_events, BASE_DATASET_REPO, args.limit)
    unbal_train_split = annotate_split("unbal_train", train_events, BASE_DATASET_REPO, args.limit)

    print(f"Merging bal_train ({len(bal_train_split)}) and unbal_train ({len(unbal_train_split)})...")
    train_split = concatenate_datasets([bal_train_split, unbal_train_split])
    
    test_split = annotate_split("eval", test_events, BASE_DATASET_REPO, args.limit)

    # 3. Finalize
    dataset = DatasetDict({"train": train_split, "test": test_split})
    print("\nFinal DatasetDict structure:")
    print(dataset)

    # 4. Push
    if not args.skip_push:
        push_to_hub(dataset, args.target_repo)

if __name__ == "__main__":
    main()