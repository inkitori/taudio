"""
Script to resample agkphysics/AudioSet dataset to 16kHz and push to enyoukai/AudioSet
"""
import os
import librosa
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, Audio
from huggingface_hub import login
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def resample_audioset_dataset(original_repo="agkphysics/AudioSet", target_repo="enyoukai/AudioSet", target_sr=16000):
    """
    Load AudioSet dataset, resample all audio to target sampling rate, and push to new repo
    
    Args:
        original_repo: Source repository name
        target_repo: Target repository name 
        target_sr: Target sampling rate
    """
    
    logger.info(f"Loading dataset from {original_repo}")
    
    # Load the original dataset
    dataset = load_dataset(original_repo, trust_remote_code=True)
    logger.info(f"Loaded dataset with splits: {list(dataset.keys())}")
    logger.info(f"Starting resampling to {target_sr} Hz")
    
    for split_name in dataset.keys():
        logger.info(f"Resampling {split_name} split...")
        
        # Update the audio feature to specify the new sampling rate
        dataset[split_name].cast_column(
            "audio", 
            Audio(sampling_rate=target_sr, mono=True, decode=True)
        )
        
        logger.info(f"Finished resampling {split_name} split")
    
    
    # Push to Hugging Face Hub
    logger.info(f"Pushing resampled dataset to {target_repo}")
    
    dataset.push_to_hub(
        target_repo,
        commit_message=f"Resample all audio to {target_sr} Hz from {original_repo}",
        private=False  # Set to True if you want a private dataset
    )
    
    logger.info(f"Successfully pushed resampled dataset to {target_repo}")

def main():
    """
    Main function to run the resampling process
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Resample AudioSet dataset to 16kHz")
    parser.add_argument("--original_repo", default="agkphysics/AudioSet", 
                       help="Source repository (default: agkphysics/AudioSet)")
    parser.add_argument("--target_repo", default="enyoukai/AudioSet",
                       help="Target repository (default: enyoukai/AudioSet)")
    parser.add_argument("--target_sr", type=int, default=16000,
                       help="Target sampling rate (default: 16000)")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="Hugging Face token for authentication")
    
    args = parser.parse_args()
    
    # Login if token provided
    if args.hf_token:
        login(token=args.hf_token)
        logger.info("Logged in to Hugging Face Hub with provided token")
    
    # Run the resampling
    resample_audioset_dataset(
        original_repo=args.original_repo,
        target_repo=args.target_repo, 
        target_sr=args.target_sr
    )

if __name__ == "__main__":
    # USAGE EXAMPLES:
    # 
    # 1. Basic usage (requires prior login with `huggingface-cli login`):
    #    python utils/audioset_resampled.py
    #
    # 2. With custom parameters:
    #    python utils/audioset_resampled.py --target_sr 22050 --target_repo "username/AudioSet-22k"
    #
    # 3. With token authentication:
    #    python utils/audioset_resampled.py --hf_token "your_hf_token_here"
    #
    # PREREQUISITES:
    # - All required dependencies are already in requirements.txt:
    #   librosa, datasets, huggingface-hub, numpy
    # - Hugging Face authentication (either via `huggingface-cli login` or --hf_token)
    # - Write access to the target repository
    #
    # The script will:
    # 1. Load the agkphysics/AudioSet dataset (both train and test splits)
    # 2. Resample all audio to 16kHz (or specified target_sr)
    # 3. Push the resampled dataset to enyoukai/AudioSet (or specified target_repo)
    # 4. Save a local backup in case of upload issues
    
    main()
