#!/usr/bin/env python3
"""
Simple script to run experiments with proper argument handling.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from config_utils import ConfigManager


def main():
    parser = argparse.ArgumentParser(description="Run TAudio experiments.")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('config', type=str, help='Config file name (without .yaml)')
    train_parser.add_argument('--no-timestamp', action='store_true',
                             help='Don\'t add timestamp to output directory')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a model')
    eval_parser.add_argument('experiment', type=str, help='Experiment name to evaluate')
    eval_parser.add_argument('--epoch', type=int, default=None,
                            help='Specific epoch to evaluate')
    eval_parser.add_argument('--split', type=str, default='dev_clean',
                            help='Dataset split to evaluate on')
    eval_parser.add_argument('--aux-output', action='store_true',
                            help='Enable auxiliary output evaluation')
    eval_parser.add_argument('--token-output', action='store_true', default=True,
                            help='Enable token output evaluation')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List experiments')
    list_parser.add_argument('--detailed', '-d', action='store_true',
                            help='Show detailed information')
    list_parser.add_argument('--filter', '-f', type=str, default=None,
                            help='Filter experiments by name')
    
    # List configs command
    configs_parser = subparsers.add_parser('configs', help='List available configs')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Check if config exists
        config_manager = ConfigManager()
        try:
            config_manager.load_config(args.config)
            print(f"Training with config: {args.config}")
        except FileNotFoundError:
            print(f"Config file not found: {args.config}.yaml")
            print("Available configs:")
            for config_file in Path("configs").glob("*.yaml"):
                print(f"  - {config_file.stem}")
            sys.exit(1)
        
        # Run training
        cmd = [sys.executable, "train.py", "--config", args.config]
        if args.no_timestamp:
            cmd.append("--no-timestamp")
        
        subprocess.run(cmd)
    
    elif args.command == 'eval':
        # Run evaluation
        cmd = [sys.executable, "evaluate.py", "--experiment", args.experiment]
        if args.epoch is not None:
            cmd.extend(["--epoch", str(args.epoch)])
        if args.split != 'dev_clean':
            cmd.extend(["--split", args.split])
        if args.aux_output:
            cmd.append("--aux-output")
        if not args.token_output:
            cmd.append("--no-token-output")
        
        subprocess.run(cmd)
    
    elif args.command == 'list':
        # List experiments
        cmd = [sys.executable, "list_experiments.py"]
        if args.detailed:
            cmd.append("--detailed")
        if args.filter:
            cmd.extend(["--filter", args.filter])
        
        subprocess.run(cmd)
    
    elif args.command == 'configs':
        # List available configs
        config_dir = Path("configs")
        if not config_dir.exists():
            print("No configs directory found.")
            return
        
        config_files = list(config_dir.glob("*.yaml"))
        if not config_files:
            print("No config files found.")
            return
        
        print(f"Available configs ({len(config_files)}):")
        for config_file in sorted(config_files):
            print(f"  - {config_file.stem}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 