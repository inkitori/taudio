"""
Utilities for loading and managing experiment configurations.
"""

import os
import yaml
import re
from typing import Dict, Any, Optional
from pathlib import Path
import datetime
from dataset import infer_adapter_from_repository


class ConfigManager:
    """Manages experiment configurations and output directories."""
    
    def __init__(self, configs_dir: str = "configs", outputs_dir: str = "outputs"):
        self.configs_dir = Path(configs_dir)
        self.outputs_dir = Path(outputs_dir)
        
        # Create directories if they don't exist
        self.configs_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load a configuration file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def create_experiment_dir(self, experiment_name: str, timestamp: bool = True) -> Path:
        """Create a directory for an experiment."""
        if timestamp:
            # Add timestamp to make each run unique
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"{experiment_name}_{timestamp_str}"
        else:
            dir_name = experiment_name
        
        experiment_dir = self.outputs_dir / dir_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        return experiment_dir
    
    def save_config(self, config: Dict[str, Any], output_dir: Path):
        """Save the configuration to the experiment directory."""
        config_path = output_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def list_experiments(self) -> list:
        """List all available experiment output directories."""
        if not self.outputs_dir.exists():
            return []
        
        return [d.name for d in self.outputs_dir.iterdir() if d.is_dir()]
    
    def get_experiment_path(self, experiment_name: str) -> Optional[Path]:
        """Get the path to an experiment directory."""
        experiment_dir = self.outputs_dir / experiment_name
        if experiment_dir.exists():
            return experiment_dir
        return None
    
    def find_latest_experiment(self, experiment_base_name: str) -> Optional[Path]:
        """Find the latest experiment with the given base name."""
        # Pattern to match exact experiment name followed by timestamp
        # Timestamp format: _YYYYMMDD_HHMMSS
        timestamp_pattern = r"_\d{8}_\d{6}$"
        expected_prefix = f"{experiment_base_name}"
        
        matching_dirs = []
        for d in self.outputs_dir.iterdir():
            if not d.is_dir():
                continue
            
            # Check if directory name starts with the expected prefix
            if not d.name.startswith(expected_prefix):
                continue
            
            # Check if the remaining part after the prefix matches the timestamp pattern
            remaining = d.name[len(expected_prefix):]
            if re.match(timestamp_pattern, remaining):
                matching_dirs.append(d)
        
        if not matching_dirs:
            return None
        
        # Sort by timestamp in directory name and return the most recent
        def extract_timestamp(dir_path):
            signature = "_YYYYMMDD_HHMMSS"
            # Extract timestamp from directory name (last 16 characters: YYYYMMDD_HHMMSS)
            timestamp_str = dir_path.name[-len(signature):]  # _YYYYMMDD_HHMMSS
            # Remove underscore and convert to comparable format: YYYYMMDDHHMMSS
            return timestamp_str.replace('_', '')
        
        latest_dir = max(matching_dirs, key=extract_timestamp)
        return latest_dir
    
    def get_model_checkpoint(self, experiment_dir: Path, epoch: Optional[int] = None) -> Optional[Path]:
        """Get the model checkpoint from an experiment directory."""
        if epoch is not None:
            checkpoint_path = experiment_dir / f"model_epoch{epoch}.pt"
        else:
            # Find the latest epoch checkpoint
            checkpoint_files = list(experiment_dir.glob("model_epoch*.pt"))
            if not checkpoint_files:
                return None
            
            # Sort by epoch number and return the highest
            checkpoint_files.sort(key=lambda x: int(x.stem.split('epoch')[1]))
            checkpoint_path = checkpoint_files[-1]
        
        if checkpoint_path.exists():
            return checkpoint_path
        return None


def flatten_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested configuration dictionary for logging."""
    flattened = {}
    
    for key, value in config.items():
        new_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(flatten_config(value, new_key))
        else:
            flattened[new_key] = value
    
    return flattened

def infer_wandb_project_from_config(config: Dict[str, Any], mode: str) -> str:
    """Infer the wandb project from the configuration."""
    return f"[{infer_adapter_from_repository(config['dataset']['repository'])}][{config['task']['type']}][{config['model']['model_id'].replace('/', '_')}]{mode}"