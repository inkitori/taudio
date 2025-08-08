import os
import shutil
import torchaudio
import torch
import random
import json
from pathlib import Path
import logging
from typing import Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_silence_to_audio(audio_path: str, output_path: str, silence_duration_min: float = 2.0, silence_duration_max: float = 10.0) -> bool:
    """
    Load an audio file, add random silence at the end, and save to output path.
    
    Args:
        audio_path: Path to the input audio file
        output_path: Path to save the modified audio file
        silence_duration_min: Minimum duration of silence in seconds
        silence_duration_max: Maximum duration of silence in seconds
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load audio with torchaudio
        audio_tensor, sampling_rate = torchaudio.load(audio_path)
        
        # Generate random silence duration between min and max
        silence_duration = random.uniform(silence_duration_min, silence_duration_max)
        silence_samples = int(silence_duration * sampling_rate)
        
        # Create silence tensor with same number of channels as original audio
        num_channels = audio_tensor.shape[0]
        silence_tensor = torch.zeros(num_channels, silence_samples)
        
        # Concatenate original audio with silence
        extended_audio = torch.cat([audio_tensor, silence_tensor], dim=1)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the extended audio
        torchaudio.save(output_path, extended_audio, sampling_rate)
        
        logger.debug(f"Added {silence_duration:.2f}s silence to {audio_path} -> {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        return False


def copy_json_files(src_dir: str, dst_dir: str) -> None:
    """
    Copy all JSON files from source directory to destination directory.
    
    Args:
        src_dir: Source directory path
        dst_dir: Destination directory path
    """
    try:
        # Ensure destination directory exists
        os.makedirs(dst_dir, exist_ok=True)
        
        # Find and copy all JSON files
        for filename in os.listdir(src_dir):
            if filename.endswith('.json'):
                src_file = os.path.join(src_dir, filename)
                dst_file = os.path.join(dst_dir, filename)
                shutil.copy2(src_file, dst_file)
                logger.debug(f"Copied {src_file} -> {dst_file}")
                
    except Exception as e:
        logger.error(f"Error copying JSON files from {src_dir} to {dst_dir}: {e}")


def process_audio_directory(src_audio_dir: str, dst_audio_dir: str, silence_duration_min: float = 2.0, silence_duration_max: float = 10.0) -> Tuple[int, int]:
    """
    Process all wav files in an audio directory by adding silence.
    
    Args:
        src_audio_dir: Source audio directory path
        dst_audio_dir: Destination audio directory path
        silence_duration_min: Minimum duration of silence in seconds
        silence_duration_max: Maximum duration of silence in seconds
        
    Returns:
        Tuple[int, int]: (successful_files, total_files)
    """
    if not os.path.exists(src_audio_dir):
        logger.warning(f"Source audio directory does not exist: {src_audio_dir}")
        return 0, 0
    
    # Ensure destination directory exists
    os.makedirs(dst_audio_dir, exist_ok=True)
    
    # Process all wav files
    wav_files = [f for f in os.listdir(src_audio_dir) if f.endswith('.wav')]
    successful = 0
    
    for wav_file in wav_files:
        src_path = os.path.join(src_audio_dir, wav_file)
        dst_path = os.path.join(dst_audio_dir, wav_file)
        
        if add_silence_to_audio(src_path, dst_path, silence_duration_min, silence_duration_max):
            successful += 1
    
    logger.info(f"Processed {successful}/{len(wav_files)} wav files in {src_audio_dir}")
    return successful, len(wav_files)


def process_dataset_split(src_split_dir: str, dst_split_dir: str, silence_duration_min: float = 2.0, silence_duration_max: float = 10.0) -> None:
    """
    Process a dataset split (train or test) by processing all subdirectories.
    
    Args:
        src_split_dir: Source split directory path
        dst_split_dir: Destination split directory path
        silence_duration_min: Minimum duration of silence in seconds
        silence_duration_max: Maximum duration of silence in seconds
    """
    if not os.path.exists(src_split_dir):
        logger.warning(f"Source split directory does not exist: {src_split_dir}")
        return
    
    logger.info(f"Processing split: {src_split_dir}")
    
    total_successful = 0
    total_files = 0
    
    # Process each subdirectory in the split
    for subdir in os.listdir(src_split_dir):
        src_subdir_path = os.path.join(src_split_dir, subdir)
        
        # Skip if not a directory
        if not os.path.isdir(src_subdir_path):
            continue
        
        dst_subdir_path = os.path.join(dst_split_dir, subdir)
        
        logger.info(f"Processing subdirectory: {subdir}")
        
        # Process audio directory if it exists
        src_audio_dir = os.path.join(src_subdir_path, "audio")
        if os.path.exists(src_audio_dir):
            dst_audio_dir = os.path.join(dst_subdir_path, "audio")
            successful, total = process_audio_directory(src_audio_dir, dst_audio_dir, silence_duration_min, silence_duration_max)
            total_successful += successful
            total_files += total
        else:
            logger.warning(f"No audio directory found in {src_subdir_path}")
        
        # Copy JSON files from subdirectory
        copy_json_files(src_subdir_path, dst_subdir_path)
    
    logger.info(f"Split processing complete: {total_successful}/{total_files} files processed successfully")


def create_audiotime_extended(
    src_dataset_dir: str,
    dst_dataset_dir: str = "AudioTimeExtended",
    silence_duration_min: float = 2.0,
    silence_duration_max: float = 10.0,
    include_test: bool = True
) -> None:
    """
    Create an extended version of the AudioTime dataset with silence added to each audio file.
    
    Args:
        src_dataset_dir: Path to the source AudioTime dataset
        dst_dataset_dir: Path for the extended dataset (default: "AudioTimeExtended")
        silence_duration_min: Minimum duration of silence to add in seconds
        silence_duration_max: Maximum duration of silence to add in seconds
        include_test: Whether to process the test split
    """
    logger.info(f"Creating AudioTimeExtended dataset from {src_dataset_dir}")
    logger.info(f"Output directory: {dst_dataset_dir}")
    logger.info(f"Silence duration range: {silence_duration_min}-{silence_duration_max} seconds")
    
    # Ensure source directory exists
    if not os.path.exists(src_dataset_dir):
        logger.error(f"Source dataset directory does not exist: {src_dataset_dir}")
        return
    
    # Create destination directory
    os.makedirs(dst_dataset_dir, exist_ok=True)
    
    # Process train split
    src_train_dir = os.path.join(src_dataset_dir, "train")
    if os.path.exists(src_train_dir):
        dst_train_dir = os.path.join(dst_dataset_dir, "train")
        process_dataset_split(src_train_dir, dst_train_dir, silence_duration_min, silence_duration_max)
    else:
        logger.warning(f"Train directory not found: {src_train_dir}")
    
    # Process test split if requested
    if include_test:
        src_test_dir = os.path.join(src_dataset_dir, "test")
        if os.path.exists(src_test_dir):
            dst_test_dir = os.path.join(dst_dataset_dir, "test")
            process_dataset_split(src_test_dir, dst_test_dir, silence_duration_min, silence_duration_max)
        else:
            logger.warning(f"Test directory not found: {src_test_dir}")
    
    logger.info("AudioTimeExtended dataset creation complete!")


def main():
    """
    Main function to create the extended AudioTime dataset.
    """
    # Configuration
    src_dataset_dir = "data/AudioTime"  # Path to your AudioTime dataset
    dst_dataset_dir = "data/AudioTimeExtended"  # Output directory
    silence_duration_min = 2.0  # Minimum silence duration in seconds
    silence_duration_max = 10.0  # Maximum silence duration in seconds
    include_test = True  # Whether to process test split
    
    # Create the extended dataset
    create_audiotime_extended(
        src_dataset_dir=src_dataset_dir,
        dst_dataset_dir=dst_dataset_dir,
        silence_duration_min=silence_duration_min,
        silence_duration_max=silence_duration_max,
        include_test=include_test
    )


if __name__ == "__main__":
    main() 