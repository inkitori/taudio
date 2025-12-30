import torch
import argparse
import logging

import torch

from accelerate import Accelerator, PartialState
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from tasks import create_task
from taudio import TAudio
from utils.config_utils import (
    ConfigManager,
)

import torch
import logging
from collections import Counter, defaultdict

def print_state_dict_detailed(state_dict):
    """
    Iterates through a model state dict, counts dtypes, calculates memory usage,
    and provides example keys for each dtype.
    """
    dtype_counts = Counter()
    dtype_bytes = defaultdict(int)
    dtype_examples = defaultdict(list)
    total_items = 0
    total_bytes = 0

    def format_size(bytes):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024.0:
                return f"{bytes:3.2f} {unit}"
            bytes /= 1024.0
        return f"{bytes:3.2f} PB"

    for key, value in state_dict.items():
        if torch.is_tensor(value):
            dt = value.dtype
            # Calculate size: number of elements * size of individual element
            size_in_bytes = value.nelement() * value.element_size()
        else:
            dt = type(value)
            size_in_bytes = 0 # Non-tensors usually contribute negligible size

        dtype_counts[dt] += 1
        dtype_bytes[dt] += size_in_bytes
        total_bytes += size_in_bytes
        total_items += 1
        
        # Keep up to 3 example keys per dtype
        if len(dtype_examples[dt]) < 3:
            dtype_examples[dt].append(key)

    logging.info("State Dict Summary:")
    logging.info(f"{'Dtype':<20} | {'Count':<8} | {'Size':<10} | {'%':<6} | {'Example Keys'}")
    logging.info("-" * 80)

    # Sort by count descending
    for dt, count in dtype_counts.most_common():
        size_str = format_size(dtype_bytes[dt])
        percent = (count / total_items) * 100
        examples = ", ".join(dtype_examples[dt])
        
        logging.info(
            f"{str(dt):<20} | {count:<8} | {size_str:<10} | {percent:>5.1f}% | {examples}"
        )

    logging.info("-" * 80)
    logging.info(f"Total Items: {total_items}")
    logging.info(f"Estimated Total Size: {format_size(total_bytes)}")

def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Unified train + distributed eval for TAudio.")
    parser.add_argument('config', type=str, help='Path to the config file')
    parser.add_argument('load', type=str, default=None, help='Path to a checkpoint to load for evaluation only')
    parser.add_argument('save', type=str, default=None, help='Path to save checkpoint')

    args = parser.parse_args()

    config_manager = ConfigManager()

    config = config_manager.load_config(f"{args.config}")

    model_config = config['model']
    loss_config = config['loss']
    task_config = config['task']
    training_config = config['training']

    task = create_task(task_type=task_config['type'], **task_config.get('kwargs', {}))

    # Build model
    taudio_config = {
        **model_config,
        **loss_config,
        "task": task
    }
    logging.info("About to create model")

    logging.info("Creating model")
    model = TAudio(**taudio_config)
    logging.info("Created model")

    # random bug where if gradients arent tracked they cause a keyerror
    # for some reason audio_bos_eos_token is also an unused param
    model.model_adapter.base_model.audio_tower.audio_bos_eos_token.requires_grad_(False)
    model.model_adapter.base_model.visual.requires_grad_(False)

    if not loss_config.get('surrogate_loss', False):
        model.linear.requires_grad_(False)

    if not loss_config.get('token_loss', False):
        model.model_adapter.base_model.lm_head.requires_grad_(False)

    accelerator = Accelerator()

    optim = torch.optim.AdamW(model.parameters(), lr=training_config['learning_rate'])

    logging.info("Using accelerator to prepare model and ghost optimizer")

    model, optim = accelerator.prepare(model, optim)
    logging.info(model.model_adapter.dtype)

    accelerator._optimizers = []

    # Accelerator load_state
    logging.info("Loading checkpoint state")
    accelerator.load_state(args.load)

    state_dict = accelerator.get_state_dict(model)

    print_state_dict_detailed(state_dict)

    logging.info("Saving checkpoint state")
    if accelerator.is_main_process:
        torch.save(state_dict, args.save)
    
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
