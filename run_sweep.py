#!/usr/bin/env python3
"""
Example script showing how to run multiple experiments with different configurations.
This demonstrates the power of the new config management system.
"""

import subprocess
import sys
import time
from pathlib import Path
import yaml


def create_sweep_configs():
    """Create configuration files for a parameter sweep."""
    
    # Base configuration
    base_config = {
        'experiment_name': 'lr_sweep',
        'description': 'Learning rate sweep experiment',
        'model': {
            'model_id': 'Qwen/Qwen2.5-Omni-3B',
            'load_in_8bit': False,
            'freeze_text_model': False,
            'audio_layer': 18
        },
        'training': {
            'epochs': 1,
            'batch_size': 1,
            'grad_accumulation_steps': 8,
            'learning_rate': 5e-6,  # Will be overridden
            'eta_min_scale': 0.1,
            'optim_8bit': True
        },
        'dataset': {
            'split': 'train_clean_100',
            'key': 'start'
        },
        'loss': {
            'class_weighting': True,
            'surrogate_loss': True,
            'token_loss': True,
            'surrogate_loss_weight': 0.2
        },
        'system': {
            'dataloader_num_workers': 8,
            'seed': 80
        },
        'wandb': {
            'entity': 'taudio',
            'project': 'Train',
            'tags': ['lr_sweep', 'experiment']
        }
    }
    
    # Different learning rates to test
    learning_rates = [1e-6, 5e-6, 1e-5, 5e-5]
    
    configs_dir = Path('configs')
    configs_dir.mkdir(exist_ok=True)
    
    config_names = []
    
    for lr in learning_rates:
        config = base_config.copy()
        config['training'] = base_config['training'].copy()  # Deep copy
        config['training']['learning_rate'] = lr
        config['experiment_name'] = f'lr_sweep_{lr:.0e}'
        config['description'] = f'Learning rate sweep with lr={lr:.0e}'
        
        config_name = f'lr_sweep_{lr:.0e}'
        config_path = configs_dir / f'{config_name}.yaml'
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        config_names.append(config_name)
        print(f"Created config: {config_path}")
    
    return config_names


def run_experiments(config_names, train_only=False):
    """Run a series of experiments."""
    
    print(f"\n{'='*60}")
    print(f"Running {len(config_names)} experiments")
    print(f"{'='*60}")
    
    experiment_dirs = []
    
    for i, config_name in enumerate(config_names, 1):
        print(f"\n[{i}/{len(config_names)}] Starting experiment: {config_name}")
        
        # Run training
        cmd = [sys.executable, 'train.py', '--config', config_name]
        print(f"Command: {' '.join(cmd)}")
        
        if train_only:
            print("DRY RUN - Would execute training command")
            # Simulate experiment directory creation
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            experiment_dir = f"{config_name}_{timestamp}"
            experiment_dirs.append(experiment_dir)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Training failed for {config_name}")
                print(f"Error: {result.stderr}")
                continue
            
            # Extract experiment directory from output (this would need to be implemented)
            # For now, we'll simulate it
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            experiment_dir = f"{config_name}_{timestamp}"
            experiment_dirs.append(experiment_dir)
            
            print(f"Training completed for {config_name}")
    
    return experiment_dirs


def run_evaluations(experiment_dirs):
    """Run evaluations for all experiments."""
    
    print(f"\n{'='*60}")
    print(f"Running evaluations for {len(experiment_dirs)} experiments")
    print(f"{'='*60}")
    
    for i, experiment_dir in enumerate(experiment_dirs, 1):
        print(f"\n[{i}/{len(experiment_dirs)}] Evaluating: {experiment_dir}")
        
        cmd = [sys.executable, 'evaluate.py', '--experiment', experiment_dir, '--aux-output']
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Evaluation failed for {experiment_dir}")
            print(f"Error: {result.stderr}")
        else:
            print(f"Evaluation completed for {experiment_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run learning rate sweep experiments")
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without actually running')
    parser.add_argument('--train-only', action='store_true',
                       help='Only run training, skip evaluation')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation (assumes experiments already exist)')
    args = parser.parse_args()
    
    if args.eval_only:
        # List existing experiments and run evaluations
        from config_utils import ConfigManager
        config_manager = ConfigManager()
        experiments = [exp for exp in config_manager.list_experiments() if 'lr_sweep' in exp]
        
        if not experiments:
            print("No lr_sweep experiments found to evaluate")
            return
        
        run_evaluations(experiments)
    else:
        # Create configs
        print("Creating sweep configurations...")
        config_names = create_sweep_configs()
        
        if args.dry_run:
            print("\nDRY RUN - Would run the following experiments:")
            for config_name in config_names:
                print(f"  - {config_name}")
            return
        
        # Run experiments
        experiment_dirs = run_experiments(config_names, train_only=args.train_only)
        
        if not args.train_only:
            # Run evaluations
            run_evaluations(experiment_dirs)
        
        print(f"\n{'='*60}")
        print("Sweep completed!")
        print(f"Ran {len(experiment_dirs)} experiments")
        print("Use 'python run_experiment.py list --detailed' to see results")
        print(f"{'='*60}")


if __name__ == "__main__":
    main() 