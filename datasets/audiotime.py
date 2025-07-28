import json
import os
import librosa
import numpy as np
import random
from datasets import Dataset, DatasetDict, Audio
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_audio_file(audio_path: str, target_sampling_rate: int = 16000) -> Dict[str, Any]:
    """
    Load audio file using librosa and return in HuggingFace Audio format.
    
    Args:
        audio_path: Path to the audio file
        target_sampling_rate: Target sampling rate for the audio
        
    Returns:
        Dictionary with audio data in HuggingFace format
    """
    try:
        # Load audio with librosa
        audio_array, sampling_rate = librosa.load(audio_path, sr=target_sampling_rate)
        
        # Ensure audio_array is a numpy array (librosa should return this, but let's be explicit)
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)
        
        # Ensure it's float32 for compatibility
        audio_array = audio_array.astype(np.float32)
        
        return {
            "path": audio_path,
            "array": audio_array,
            "sampling_rate": sampling_rate
        }
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}")
        return None


def extend_audio_with_noise(example, min_duration=5.0, max_duration=15.0):
    """
    Extend audio with random white noise for 5-15 seconds.
    
    Args:
        example: Dataset example containing audio data
        min_duration: Minimum duration to add in seconds
        max_duration: Maximum duration to add in seconds
        
    Returns:
        Modified example with extended audio
    """
    # Get audio data
    audio_data = example["audio"]
    audio_array = audio_data["array"]
    sampling_rate = audio_data["sampling_rate"]
    
    # Randomly choose extension duration between min_duration and max_duration
    extension_duration = random.uniform(min_duration, max_duration)
    extension_samples = int(extension_duration * sampling_rate)
    
    # Generate white noise with same amplitude range as original audio
    audio_std = np.std(audio_array) if len(audio_array) > 0 else 0.1
    noise = np.random.normal(0, audio_std * 0.1, extension_samples).astype(np.float32)
    
    # Concatenate original audio with noise
    extended_audio = np.concatenate([audio_array, noise])
    example['audio']['array'] = extended_audio
    
    return example


def parse_timestamp_events(events: Dict[str, List[List[float]]]) -> List[Dict[str, Any]]:
    """
    Parse timestamp events from JSON format to the desired word format.
    
    Args:
        events: Dictionary with event names as keys and list of [start, end] timestamps as values
        
    Returns:
        List of dictionaries with 'word', 'start', 'end' keys, ordered by start time
    """
    words = []
    
    # Extract all events with their timestamps
    for event_name, timestamp_pairs in events.items():
        for start_time, end_time in timestamp_pairs:
            words.append({
                "word": event_name,
                "start": start_time,
                "end": end_time
            })
    
    # Sort by start time to maintain temporal order
    words.sort(key=lambda x: x["start"])
    
    return words


def parse_audiotime_split(split_dir: str, target_sampling_rate: int = 16000, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Parse a single split (train/test) of the AudioTime dataset.
    
    Args:
        split_dir: Path to the split directory (e.g., train/ or test/)
        target_sampling_rate: Target sampling rate for audio processing
        max_examples: Maximum number of examples to load (None for unlimited)
        
    Returns:
        List of examples for the dataset
    """
    examples = []
    
    # Find all subdirectories in the split
    if not os.path.exists(split_dir):
        logger.warning(f"Split directory {split_dir} does not exist")
        return examples
    
    for subdir in os.listdir(split_dir):
        subdir_path = os.path.join(split_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        audio_dir = os.path.join(subdir_path, "audio")
        caption_file = os.path.join(subdir_path, "timestamp_captions.json")
        
        # Check if required files exist
        if not os.path.exists(audio_dir) or not os.path.exists(caption_file):
            logger.warning(f"Missing audio directory or caption file in {subdir_path}")
            continue
        
        # Load timestamp captions
        try:
            with open(caption_file, 'r') as f:
                captions_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading caption file {caption_file}: {e}")
            continue
        
        # Process each audio file
        for audio_filename in os.listdir(audio_dir):
            if not audio_filename.endswith('.wav'):
                continue
                
            # Extract the key (e.g., 'syn_1' from 'syn_1.wav')
            audio_key = os.path.splitext(audio_filename)[0]
            
            if audio_key not in captions_data:
                logger.warning(f"No caption data found for {audio_key}")
                continue
            
            audio_path = os.path.join(audio_dir, audio_filename)
            
            
            # Parse timestamp events
            caption_info = captions_data[audio_key]
            events = caption_info.get("event", {})
            words = parse_timestamp_events(events)
            
            # Create example
            example = {
                "audio": audio_path,
                "words": words,
                "caption": caption_info.get("caption", ""),
                "audio_id": audio_key,
                "split": os.path.basename(split_dir)
            }
            
            examples.append(example)
            
            # Check if we've reached the maximum number of examples
            if max_examples is not None and len(examples) >= max_examples:
                logger.info(f"Reached maximum number of examples ({max_examples}), stopping processing")
                return examples
            
        logger.info(f"Processed {len(examples)} examples from {subdir_path}")
    
    return examples


def create_audiotime_dataset(
    data_dir: str, 
    target_sampling_rate: int = 16000,
    include_test: bool = True,
    max_examples: Optional[int] = None
) -> DatasetDict:
    """
    Create a HuggingFace DatasetDict from the AudioTime dataset.
    
    Args:
        data_dir: Path to the AudioTime dataset directory
        target_sampling_rate: Target sampling rate for audio processing
        include_test: Whether to include test split
        max_examples: Maximum number of examples to load per split (None for unlimited)
        
    Returns:
        HuggingFace DatasetDict with train (and optionally test) splits
    """
    dataset_dict = {}
    
    # Process train split
    train_dir = os.path.join(data_dir, "train")
    logger.info(f"Processing train split from {train_dir}")
    train_examples = parse_audiotime_split(train_dir, target_sampling_rate, max_examples)
    
    if train_examples:
        train_dataset = Dataset.from_list(train_examples)
        # Cast audio column to Audio feature for proper handling
        train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))
        # Extend audio with random noise (5-15 seconds)
        logger.info("Extending train audio with random noise...")
        train_dataset = train_dataset.map(extend_audio_with_noise, desc="Extending audio")
        dataset_dict["train"] = train_dataset
        logger.info(f"Created train split with {len(train_examples)} examples")
    else:
        logger.warning("No train examples found")
    
    # Process test split if requested
    if include_test:
        test_dir = os.path.join(data_dir, "test")
        logger.info(f"Processing test split from {test_dir}")
        test_examples = parse_audiotime_split(test_dir, target_sampling_rate, max_examples)
        
        if test_examples:
            test_dataset = Dataset.from_list(test_examples)
            # Cast audio column to Audio feature for proper handling
            test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))
            # Extend audio with random noise (5-15 seconds)
            logger.info("Extending test audio with random noise...")
            test_dataset = test_dataset.map(extend_audio_with_noise, desc="Extending audio")
            dataset_dict["test"] = test_dataset
            logger.info(f"Created test split with {len(test_examples)} examples")
        else:
            logger.warning("No test examples found")
    
    return DatasetDict(dataset_dict)


def save_dataset(dataset: DatasetDict, output_dir: str):
    """
    Save the dataset to disk.
    
    Args:
        dataset: HuggingFace DatasetDict to save
        output_dir: Directory to save the dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    logger.info(f"Dataset saved to {output_dir}")


def main():
    """
    Main function to parse AudioTime dataset.
    Example usage of the parsing functions.
    """
    # Configuration
    data_dir = "/anvil/projects/ai250124/x-pkeung/anjo0/taudio/data/AudioTime"  # Path to your AudioTime dataset
    output_dir = "audiotime_hf_dataset"  # Output directory for processed dataset
    target_sampling_rate = 16000  # Target sampling rate
    max_examples = None  # Maximum number of examples per split (None for unlimited)
    
    logger.info("Starting AudioTime dataset parsing...")
    
    # Create dataset
    dataset = create_audiotime_dataset(
        data_dir=data_dir,
        target_sampling_rate=target_sampling_rate,
        include_test=True,
        max_examples=max_examples
    )
    
    # Print dataset info
    if dataset:
        logger.info("Dataset created successfully!")
        for split_name, split_dataset in dataset.items():
            logger.info(f"{split_name} split: {len(split_dataset)} examples")
            
            # Print example structure
            if len(split_dataset) > 0:
                example = split_dataset[0]
                logger.info(f"Example structure for {split_name} split:")
                logger.info(f"  - audio: {type(example['audio'])}")
                logger.info(f"  - words: {len(example['words'])} events")
                if example['words']:
                    logger.info(f"    First word: {example['words'][0]}")
                logger.info(f"  - caption: {example['caption'][:100]}...")
                logger.info(f"  - audio_id: {example['audio_id']}")
        
        # Save dataset
        # save_dataset(dataset, output_dir)
        
        # Optionally push to HuggingFace Hub
        dataset.push_to_hub("enyoukai/audiotime-timestamps")
        
    else:
        logger.error("Failed to create dataset")


if __name__ == "__main__":
    main()
