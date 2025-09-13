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
    
    def create_experiment_dir(self, config_path: str, timestamp: bool = True) -> Path:
        """Create a directory for an experiment based on config path structure."""
        # Convert config path to experiment directory path
        # Replace "configs/" with "outputs/" and remove .yaml extension
        config_path = Path(config_path)
        
        # Remove the "configs/" prefix and .yaml extension
        relative_path = config_path.relative_to(self.configs_dir) if str(config_path).startswith("configs/") else config_path
        experiment_subpath = relative_path.with_suffix("")  # Remove .yaml extension
        
        if timestamp:
            # Add timestamp to make each run unique
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_dir = self.outputs_dir / experiment_subpath / timestamp_str
        else:
            experiment_dir = self.outputs_dir / experiment_subpath
        
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        return experiment_dir
    
    def save_config(self, config: Dict[str, Any], experiment_dir: Path):
        """Save the configuration to the experiment directory."""
        config_path = experiment_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
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


def relative_path_to_experiment_name(config_path: str) -> str:
    """
    Convert a relative path to an experiment name.
    should be prepended with "configs/"
    example: configs/qwen3b/librispeech/timestamps/bernoulli/bidirectional_audio.yaml -> bidirectional_audio
    """
    config_path_parts = config_path.split("/")
    filename = Path(config_path_parts[-1]).stem  # Remove file extension using pathlib
    
    return filename

def relative_path_to_project_name(config_path: str, mode: str) -> str:
    """Convert a relative path to a project name."""
    config_path_parts = config_path.split("/")
    directories = config_path_parts[1:-1]  # All parts except filename and configs/
    
    # Format as [dir1][dir2][dir3]
    return "".join(f"[{dir}]" for dir in directories) + f"[{mode}]"