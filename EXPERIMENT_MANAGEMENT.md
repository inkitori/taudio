# TAudio Experiment Management

This document describes the refactored experiment management system for TAudio.

## Overview

The new system provides:
- **Clean configuration management** with YAML files
- **Organized experiment outputs** in structured directories
- **Meaningful wandb run names** based on experiment configs
- **Easy experiment tracking** and evaluation
- **Automated model/config loading** for evaluation

## Directory Structure

```
taudio/
├── configs/                    # Experiment configuration files
│   ├── base_experiment.yaml
│   ├── end_prediction.yaml
│   └── high_lr_experiment.yaml
├── outputs/                    # Experiment outputs (auto-created)
│   ├── base_experiment_20241201_143022/
│   │   ├── config.yaml        # Copy of config used
│   │   ├── model_epoch1.pt    # Model checkpoint
│   │   └── model_epoch2.pt
│   └── end_prediction_20241201_150045/
│       ├── config.yaml
│       └── model_epoch1.pt
├── train.py                   # Training script
├── evaluate.py               # Evaluation script
├── run_experiment.py         # Main experiment runner
├── list_experiments.py       # Utility to list experiments
└── config_utils.py           # Configuration management utilities
```

## Configuration Files

Configuration files are organized into logical sections:

```yaml
# configs/my_experiment.yaml
experiment_name: "my_experiment"
description: "Description of what this experiment tests"

# Model configuration
model:
  model_id: "Qwen/Qwen2.5-Omni-3B"
  load_in_8bit: false
  freeze_text_model: false
  audio_layer: 18

# Training configuration
training:
  epochs: 1
  batch_size: 1
  grad_accumulation_steps: 8
  learning_rate: 5e-6
  eta_min_scale: 0.1
  optim_8bit: true

# Dataset configuration
dataset:
  split: "train_clean_100"
  key: "start"  # or "end"

# Loss configuration
loss:
  class_weighting: true
  surrogate_loss: true
  token_loss: true
  surrogate_loss_weight: 0.2

# System configuration
system:
  dataloader_num_workers: 8
  seed: 80

# Wandb configuration
wandb:
  entity: "taudio"
  project: "Train"
  tags: ["experiment", "tag1", "tag2"]
```

## Usage

### 1. Running Experiments

#### Simple approach:
```bash
# Train with a config
python run_experiment.py train base_experiment

# Evaluate an experiment
python run_experiment.py eval base_experiment_20241201_143022

# List all experiments
python run_experiment.py list --detailed
```

#### Direct approach:
```bash
# Train
python train.py --config base_experiment

# Evaluate
python evaluate.py --experiment base_experiment_20241201_143022 --aux-output
```

### 2. Managing Experiments

#### List available configs:
```bash
python run_experiment.py configs
```

#### List experiments:
```bash
# Simple list
python run_experiment.py list

# Detailed list with config info
python run_experiment.py list --detailed

# Filter by name
python run_experiment.py list --filter "base_experiment"
```

#### Direct listing:
```bash
python list_experiments.py --detailed --filter "end_prediction"
```

### 3. Evaluation

The evaluation script automatically finds the right config and checkpoint:

```bash
# Evaluate latest epoch
python evaluate.py --experiment base_experiment_20241201_143022

# Evaluate specific epoch
python evaluate.py --experiment base_experiment_20241201_143022 --epoch 1

# Evaluate with auxiliary output
python evaluate.py --experiment base_experiment_20241201_143022 --aux-output

# Evaluate on different split
python evaluate.py --experiment base_experiment_20241201_143022 --split dev_other
```

## Key Features

### 1. Automatic Directory Management
- Experiments are automatically organized in timestamped directories
- Configs are saved with each experiment for reproducibility
- Checkpoints are clearly named by epoch

### 2. Meaningful Wandb Names
Wandb run names are automatically generated based on key config parameters:
- `base_experiment_key-start_lr-5e-06`
- `end_prediction_key-end_lr-5e-06`
- `high_lr_experiment_key-start_lr-1e-05_frozen`

### 3. Flexible Configuration
- Easy to create new experiments by copying/modifying config files
- All parameters are in one place, not scattered through code
- Hierarchical config structure for better organization

### 4. Smart Experiment Finding
- Evaluation can find experiments by exact name or base name
- Automatically finds the latest checkpoint if no epoch specified
- Lists available experiments if the target isn't found

## Creating New Experiments

1. **Copy an existing config:**
   ```bash
   cp configs/base_experiment.yaml configs/my_new_experiment.yaml
   ```

2. **Modify the config as needed:**
   - Change `experiment_name` and `description`
   - Adjust any parameters you want to test
   - Add appropriate tags

3. **Run the experiment:**
   ```bash
   python run_experiment.py train my_new_experiment
   ```

4. **Evaluate when done:**
   ```bash
   python run_experiment.py eval my_new_experiment
   ```

## Migration from Old System

If you have existing checkpoints in `data/checkpoints/`, you can:

1. Create a config file that matches the old parameters
2. Move the checkpoint to the new structure:
   ```bash
   mkdir -p outputs/legacy_experiment_20241201_000000
   cp data/checkpoints/model_xyz_epoch1.pt outputs/legacy_experiment_20241201_000000/model_epoch1.pt
   cp data/checkpoints/model_xyz.yml outputs/legacy_experiment_20241201_000000/config.yaml
   ```

## Benefits

- **No more random wandb IDs** - meaningful experiment names
- **No more manual path management** - everything is organized automatically
- **Easy parameter sweeps** - just create new config files
- **Reproducible experiments** - configs are saved with each run
- **Simple evaluation** - no need to manually specify paths
- **Better experiment tracking** - clear directory structure and naming 