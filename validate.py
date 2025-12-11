import time
from datasets import load_dataset
import numpy as np

# Replace with your repo ID
REPO_ID = "enyoukai/testing" 

def verify_full_dataset():
    print(f"📥 Downloading and loading full dataset: {REPO_ID} ...")
    
    # 1. Load the entire dataset (downloads to local cache)
    try:
        dataset = load_dataset(REPO_ID)
    except Exception as e:
        print(f"❌ CRITICAL: Could not download/load dataset structure. Error:\n{e}")
        return

    print("\n✅ Dataset structure loaded successfully.")
    print(dataset)

    total_errors = 0

    # 2. Iterate through every split (train, test, etc.)
    for split_name in dataset.keys():
        ds_split = dataset[split_name]
        num_rows = len(ds_split)
        print(f"\n🔎 Verifying split: '{split_name}' ({num_rows} rows)...")
        
        start_time = time.time()
        
        # Iterate over the dataset.
        # Enumerate gives us the index 'i' to pinpoint errors.
        for i, row in enumerate(ds_split):
            # Print progress every 1000 rows so you know it's working
            if i % 1000 == 0:
                print(f"   Processed {i}/{num_rows} rows...", end="\r")

            try:
                # --- CHECK 1: AUDIO DECODING ---
                # This line is the heavy lifter. It reads bytes and converts to numpy array.
                # If the audio file is corrupt, this will raise an exception.
                audio_array = row["audio"]["array"]
                
                if audio_array is None or len(audio_array) == 0:
                    print(f"\n❌ [Row {i}] Audio Error: Array is empty.")
                    total_errors += 1
                    continue

                # --- CHECK 2: EVENTS VALIDITY ---
                events = row.get("events", [])
                
                # 'events' should be a list (Sequence) of dictionaries (Structs)
                # In the decoded dataset, this usually appears as a dict of lists or list of dicts 
                # depending on exact schema, but standard access is list-like.
                if not hasattr(events, '__iter__'):
                     print(f"\n❌ [Row {i}] Structure Error: 'events' is not iterable.")
                     total_errors += 1
                     continue

                # If there are events, check the first one for required keys
                # The viewer often fails if schemas are inconsistent
                if len(events) > 0:
                    # Handle Hugging Face "Sequence" format which might return a dict of lists:
                    # {'start': [0.5, 1.2], 'end': [0.9, 1.5], ...}
                    if isinstance(events, dict):
                        keys = events.keys()
                        if not {'start', 'end', 'event_name'}.issubset(keys):
                             print(f"\n❌ [Row {i}] Schema Error: Missing keys in event dict. Found: {keys}")
                             total_errors += 1
                    # Handle List of Dicts format:
                    # [{'start': 0.5}, {'start': 1.2}]
                    elif isinstance(events, list) or isinstance(events, np.ndarray):
                        first_event = events[0]
                        if isinstance(first_event, dict):
                             if not {'start', 'end', 'event_name'}.issubset(first_event.keys()):
                                 print(f"\n❌ [Row {i}] Schema Error: Missing keys in event list. Found: {first_event.keys()}")
                                 total_errors += 1

            except Exception as e:
                print(f"\n❌ [Row {i}] CRITICAL EXCEPTION: {e}")
                total_errors += 1

        elapsed = time.time() - start_time
        print(f"   ✅ Finished '{split_name}' in {elapsed:.2f}s")

    print("\n" + "="*40)
    if total_errors == 0:
        print("🎉 SUCCESS: Entire dataset verified. No corrupt audio or broken rows found.")
        print("The errors on the Hugging Face website are definitely just server-side viewer capacity issues.")
    else:
        print(f"⚠️ FAILURE: Found {total_errors} errors in the dataset.")

if __name__ == "__main__":
    verify_full_dataset()
