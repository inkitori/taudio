#!/usr/bin/env python3
"""
Utility script to list all available experiments and their details.
"""

import argparse
from pathlib import Path
import yaml
from datetime import datetime

from config_utils import ConfigManager


def format_file_size(size_bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"


def main():
    parser = argparse.ArgumentParser(description="List available experiments.")
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Show detailed information about each experiment')
    parser.add_argument('--filter', '-f', type=str, default=None,
                       help='Filter experiments by name (substring match)')
    args = parser.parse_args()
    
    config_manager = ConfigManager()
    experiments = config_manager.list_experiments()
    
    if args.filter:
        experiments = [exp for exp in experiments if args.filter.lower() in exp.lower()]
    
    if not experiments:
        print("No experiments found.")
        return
    
    print(f"Found {len(experiments)} experiment(s):")
    print("=" * 80)
    
    for exp_name in sorted(experiments):
        exp_path = config_manager.get_experiment_path(exp_name)
        
        if not args.detailed:
            print(f"  {exp_name}")
            continue
        
        # Detailed view
        print(f"\nExperiment: {exp_name}")
        print(f"Path: {exp_path}")
        
        # Get creation time
        creation_time = datetime.fromtimestamp(exp_path.stat().st_ctime)
        print(f"Created: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load and display config
        config_path = exp_path / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                print(f"Description: {config.get('description', 'N/A')}")
                print(f"Model: {config.get('model', {}).get('model_id', 'N/A')}")
                print(f"Dataset: {config.get('dataset', {}).get('split', 'N/A')}")
                print(f"Key: {config.get('dataset', {}).get('key', 'N/A')}")
                print(f"Learning Rate: {config.get('training', {}).get('learning_rate', 'N/A')}")
                print(f"Epochs: {config.get('training', {}).get('epochs', 'N/A')}")
                
            except Exception as e:
                print(f"Error loading config: {e}")
        
        # List checkpoints
        checkpoint_files = list(exp_path.glob("model_epoch*.pt"))
        if checkpoint_files:
            print(f"Checkpoints ({len(checkpoint_files)}):")
            for checkpoint in sorted(checkpoint_files):
                size = format_file_size(checkpoint.stat().st_size)
                epoch = checkpoint.stem.split('epoch')[1]
                print(f"  - Epoch {epoch}: {size}")
        else:
            print("No checkpoints found")
        
        print("-" * 80)


if __name__ == "__main__":
    main() 