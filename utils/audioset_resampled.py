#!/usr/bin/env python3
"""
Script to pull AudioSet dataset from HuggingFace, remove specific indices, 
and push to a new repository.

This script:
1. Downloads the agkphysics/AudioSet dataset
2. Removes specified train indices (15,759 and 17,532) and test index (6,182)
3. Pushes the modified dataset to enyoukai/AudioSet
"""

import os
import logging
from datasets import load_dataset, DatasetDict
from huggingface_hub import login, HfApi
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def remove_indices_from_split(dataset, indices_to_remove: List[int], split_name: str):
    """
    Remove specific indices from a dataset split.
    
    Args:
        dataset: The dataset split to modify
        indices_to_remove: List of indices to remove
        split_name: Name of the split (for logging)
    
    Returns:
        Modified dataset with specified indices removed
    """
    logger.info(f"Original {split_name} dataset size: {len(dataset)}")
    
    return dataset.select(
        (i for i in range(len(dataset)) if i not in indices_to_remove)
    )

def main():
    """Main function to process and upload the dataset."""
    
    # Configuration
    source_dataset = "agkphysics/AudioSet"
    target_dataset = "enyoukai/AudioSet"
    
    # Indices to remove
    train_indices_to_remove = [15759, 17532]  # Note: using 0-based indexing
    test_indices_to_remove = [6182]
    
    try:
        # Step 1: Load the original dataset
        logger.info(f"Loading dataset from {source_dataset}...")
        dataset = load_dataset(source_dataset, trust_remote_code=True)
        logger.info(f"Dataset loaded successfully. Splits: {list(dataset.keys())}")
        
        # Log original sizes
        for split_name in dataset.keys():
            logger.info(f"Original {split_name} size: {len(dataset[split_name])}")
        
        # Step 2: Process each split
        modified_dataset = {}
        
        # Process train split
        if 'train' in dataset:
            modified_dataset['train'] = remove_indices_from_split(
                dataset['train'], 
                train_indices_to_remove, 
                'train'
            )
        
        # Process test split
        if 'test' in dataset:
            modified_dataset['test'] = remove_indices_from_split(
                dataset['test'], 
                test_indices_to_remove, 
                'test'
            )
        
        # Copy other splits without modification
        for split_name in dataset.keys():
            if split_name not in ['train', 'test']:
                logger.info(f"Copying {split_name} split without modification")
                modified_dataset[split_name] = dataset[split_name]
        
        # Create DatasetDict
        final_dataset = DatasetDict(modified_dataset)
        
        # # Step 3: Login to Hugging Face (requires token)
        # logger.info("Logging in to Hugging Face...")
        # try:
        #     login()  # This will use the token from environment or prompt for it
        # except Exception as e:
        #     logger.error(f"Failed to login to Hugging Face: {e}")
        #     logger.info("Please make sure you have a valid HF token set via 'huggingface-cli login' or HF_TOKEN environment variable")
        #     return
        
        # Step 4: Push to new repository
        logger.info(f"Pushing modified dataset to {target_dataset}...")
        final_dataset.push_to_hub(
            target_dataset,
            private=False,  # Set to True if you want a private repository
            commit_message=f"Modified AudioSet: removed train indices {train_indices_to_remove} and test indices {test_indices_to_remove}"
        )
        
        logger.info("Dataset upload completed successfully!")
        
        # Log final statistics
        logger.info("Final dataset statistics:")
        for split_name, split_data in final_dataset.items():
            logger.info(f"  {split_name}: {len(split_data)} samples")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
