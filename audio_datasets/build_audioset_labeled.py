import pandas as pd
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder
import pandas as pd
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import os

def create_dataset_split(csv_path, dataset_name, split):
    """
    Processes a single CSV file and a corresponding dataset split to create a new labeled dataset split using a much faster filtering method.

    Args:
        csv_path (str): The file path to the CSV containing event information.
        dataset_name (str): The name of the Hugging Face dataset to load (e.g., 'enyoukai/AudioSet').
        split (str): The dataset split to use (e.g., 'train', 'test').

    Returns:
        Dataset: The new, filtered dataset split with an added 'events' field, or None if an error occurs.
    """
    print(f"\n--- Processing split: '{split}' using file: '{csv_path}' ---")

    # 1. Load the CSV file
    try:
        events_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file at {csv_path} was not found.")
        return None

    # 2. Get a set of unique video_ids from the CSV for efficient filtering
    video_ids_with_events = set(events_df['id'].unique())
    print(f"Found {len(video_ids_with_events)} unique video IDs with events in '{csv_path}'.")

    # 3. Group events by 'id' into a dictionary for fast lookup
    events_by_video_id = {}
    for video_id, group in events_df.groupby('id'):
        events_list = []
        for _, row in group.iterrows():
            event = {
                'event_name': row['label_name'],
                'start': row['label_start'] - row['start_time'],
                'end': row['label_end'] - row['start_time']
            }
            events_list.append(event)
        events_by_video_id[video_id] = events_list

    # 4. Load the specified split from the Hugging Face dataset
    original_dataset_split = load_dataset(dataset_name, name='unbalanced', split=split, trust_remote_code=True)
    print(f"Original '{split}' split size: {len(original_dataset_split)} examples.")

    # 5. (OPTIMIZED) Find indices to keep instead of filtering directly
    print("Finding indices of matching video_ids...")
    indices_to_keep = [
        i for i, video_id in enumerate(tqdm(original_dataset_split['video_id']))
        if video_id in video_ids_with_events
    ]

    if not indices_to_keep:
        print(f"Warning: No matching video_ids found for the '{split}' split. Check your CSV and split name.")
        return None

    # 6. (OPTIMIZED) Use .select() which is much faster as it doesn't rewrite the dataset
    filtered_split = original_dataset_split.select(indices_to_keep)
    print(f"Filtered '{split}' split size: {len(filtered_split)} examples.")


    # 7. Map the events data to the filtered split
    def add_events_to_example(example):
        video_id = example['video_id']
        example['events'] = events_by_video_id[video_id]
        return example

    # For potentially faster mapping, you can add num_proc to use multiple processes
    processed_split = filtered_split.map(add_events_to_example, num_proc=4) # Adjust num_proc based on your CPU cores
    return processed_split

def push_dataset_to_hub(dataset_dict, repo_name):
    """
    Pushes a Hugging Face DatasetDict to the Hub.

    Args:
        dataset_dict (DatasetDict): The dataset dictionary to push.
        repo_name (str): The name of the repository on the Hugging Face Hub
                         (e.g., 'username/dataset-name').
    """
    try:
        print(f"\nPushing dataset to Hugging Face Hub at '{repo_name}'...")
        # The push_to_hub method works directly on a DatasetDict object
        dataset_dict.push_to_hub(repo_name, max_shard_size="2GB")
        print("Successfully pushed dataset to the Hub!")
    except Exception as e:
        print(f"An error occurred while pushing the dataset: {e}")

if __name__ == '__main__':
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # --- Configuration ---
    TRAIN_CSV_PATH = 'resampled_train_set.csv'
    EVAL_CSV_PATH = 'merged_eval.csv'
    DATASET_NAME = 'agkphysics/AudioSet'
    # IMPORTANT: Replace with your Hugging Face username and desired dataset name.
    HF_REPO_NAME = "enyoukai/AudioSet-Strong"

    # --- Step 1: Create the training split ---
    train_split = create_dataset_split(
        csv_path=TRAIN_CSV_PATH,
        dataset_name=DATASET_NAME,
        split='train'
    )

    # --- Step 2: Create the evaluation split ---
    # NOTE: enyoukai/AudioSet uses 'test' for its second split.
    eval_split = create_dataset_split(
        csv_path=EVAL_CSV_PATH,
        dataset_name=DATASET_NAME,
        split='test' # Using the 'test' split for evaluation
    )

    # --- Step 3: Combine splits into a DatasetDict and push ---
    if train_split and eval_split:
        # Create the DatasetDict
        final_dataset = DatasetDict({
            'train': train_split,
            'eval': eval_split
        })

        print("\n--- Dataset creation complete ---")
        print(final_dataset)
        print("\nExample from 'train' split:")
        print(final_dataset['train'][0])
        print("\nExample from 'eval' split:")
        print(final_dataset['eval'][0])


        # Push to Hub
        if "your-hf-username" in HF_REPO_NAME:
            print("\nWARNING: Please update the 'HF_REPO_NAME' variable with your actual Hugging Face username and a dataset name before pushing.")
        else:
            push_dataset_to_hub(final_dataset, HF_REPO_NAME)
    else:
        print("\nCould not create one or both dataset splits. Please check the logs for errors.")
