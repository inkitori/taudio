"""Utility for pushing the LibriCount audio mixtures to the Hugging Face Hub.

This script reads the `mixtures_metadata.json` file that accompanies the
LibriCount mixtures and constructs a Hugging Face dataset with the following
schema:

    - audio: the mixture audio file
    - k: the number of speakers in the mixture
    - components: metadata for each component utterance sorted by
      `first_word_start_sec`

The resulting dataset is uploaded to the Hub repository specified by
`--repo-id` (defaults to `enyoukai/libricount-timings`).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from datasets import Audio, Dataset, DatasetDict, Features, Sequence, Value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parent
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=default_root / "mixtures_start_0" / "mixtures_metadata.json",
        help="Path to the LibriCount mixtures metadata JSON file.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=default_root,
        help=(
            "Directory that acts as the root for relative audio paths in the "
            "metadata. Defaults to the directory containing this script."
        ),
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="enyoukai/libricount-timings",
        help="Target Hugging Face Hub repository ID (e.g. 'user/dataset').",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Name of the dataset split to create.",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="2GB",
        help="Maximum shard size used when pushing the dataset to the Hub.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HF_TOKEN"),
        help=(
            "Hugging Face access token. Defaults to the value in the "
            "HUGGINGFACEHUB_API_TOKEN or HF_TOKEN environment variables if "
            "present."
        ),
    )
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> List[Dict[str, object]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sort_components(components: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        (
            {
                "speaker_id": str(component["speaker_id"]),
                "utterance_id": str(component["utterance_id"]),
                "utterance_length_sec": float(component["utterance_length_sec"]),
                "first_word_start_sec": float(component["first_word_start_sec"]),
            }
            for component in components
        ),
        key=lambda item: item["first_word_start_sec"],
    )


def make_features() -> Features:
    return Features(
        {
            "audio": Audio(),
            "k": Value("int32"),
            "components": Sequence(
                {
                    "speaker_id": Value("string"),
                    "utterance_id": Value("string"),
                    "utterance_length_sec": Value("float32"),
                    "first_word_start_sec": Value("float32"),
                }
            ),
        }
    )


def dataset_generator(
    metadata: List[Dict[str, object]],
    audio_root: Path,
) -> Iterator[Dict[str, object]]:
    for record in metadata:
        mixture_path = Path(record["mixture_path"])
        if not mixture_path.is_absolute():
            mixture_path = audio_root / mixture_path

        if not mixture_path.exists():
            raise FileNotFoundError(f"Audio file referenced in metadata not found: {mixture_path}")

        yield {
            "audio": str(mixture_path),
            "k": int(record["k"]),
            "components": sort_components(record["components"]),
        }


def build_dataset(metadata_path: Path, audio_root: Path) -> Dataset:
    metadata = load_metadata(metadata_path)

    def generator() -> Iterator[Dict[str, object]]:
        return dataset_generator(metadata, audio_root)

    dataset = Dataset.from_generator(generator)
    dataset = dataset.cast_column("audio", Audio())

    return dataset


def push_dataset(dataset: Dataset, repo_id: str, split: str, max_shard_size: str, token: str | None) -> None:
    dataset_dict = DatasetDict({split: dataset})
    dataset_dict.push_to_hub(repo_id, token=token, max_shard_size=max_shard_size)


def main() -> None:
    args = parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    dataset = build_dataset(args.metadata_path, args.audio_root)
    push_dataset(dataset, args.repo_id, args.split, args.max_shard_size, args.token)


if __name__ == "__main__":
    main()



