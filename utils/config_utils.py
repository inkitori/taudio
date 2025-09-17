"""
Utilities for loading and managing experiment configurations.
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path


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
    
    def create_experiment_dir(self, config_path: str) -> Path:
        """Create a directory for an experiment based on config path structure."""
        # Convert config path to experiment directory path
        # Replace "configs/" with "outputs/" and remove .yaml extension
        config_path = Path(config_path)
        
        # Remove the "configs/" prefix and .yaml extension
        relative_path = config_path.relative_to(self.configs_dir) if str(config_path).startswith("configs/") else config_path
        experiment_subpath = relative_path.with_suffix("")  # Remove .yaml extension
        
        experiment_dir = self.outputs_dir / experiment_subpath
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        return experiment_dir
    
    def save_config(self, config: Dict[str, Any], experiment_dir: Path):
        """Save the configuration to the experiment directory."""
        config_path = experiment_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def get_model_checkpoint(self, experiment_dir: Path, epoch: Optional[int] = None) -> Optional[Path]:
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


def relative_path_to_experiment_name(config_path: str, eval: bool) -> str:
    """
    Convert a relative path to an experiment name.
    If eval is True only pass the experiment directory
    
    Train example:
    configs/qwen3b/librispeech/timestamps/bernoulli/bidirectional_audio.yaml -> bidirectional_audio

    Eval example:
    outputs/qwen7b/librispeech/timestamp_single/bernoulli+bidirectional_audio+class_weighting[start][float16]/20250913_175143 -> 'bernoulli+bidirectional_audio+class_weighting[start][float16]_20250913_175143'
    """
    config_path_parts = config_path.split("/")
    if eval:
        return config_path_parts[-2] + '_' + config_path_parts[-1] # returns the experiment name + timestamp
    else:
        filename = Path(config_path_parts[-1]).stem  # Remove file extension using pathlib
        
        return filename

def relative_path_to_project_name(config_path: str, eval: bool) -> str:
    """
    Convert a relative path to a project name.
    If eval is True only pass the experiment directory

	Train example:
    configs/qwen3b/librispeech/timestamps/bernoulli/bidirectional_audio.yaml -> '[qwen3b][librispeech][timestamps][Train]'

    Eval example:
    outputs/qwen7b/librispeech/timestamp_single/bernoulli+bidirectional_audio+class_weighting[start][float16]/20250913_175143 -> '[qwen7b][librispeech][timestamp_single][Eval]'
    """
    config_path_parts = config_path.split("/")
    if eval:
        directories = config_path_parts[1:-2] # Get rid of timestamp and experiment name
    else:
        directories = config_path_parts[1:-1]  # All parts except filename and configs/

    # Format as [dir1][dir2][dir3]
    return "".join(f"[{dir}]" for dir in directories) + f"[{"Eval" if eval else "Train"}]"