import time
import torch
from collections import Counter


import argparse
import logging
import os
import random
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
from transformers import set_seed

from accelerate import Accelerator, PartialState
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from dataset.dataset import collate_fn, get_ds
from dataset import create_adapter, infer_adapter_from_repository
from tasks import create_task
from taudio import TAudio
from utils.config_utils import (
    ConfigManager,
)

def print_gpu_memory():
    if not torch.cuda.is_available():
        logging.info("No CUDA GPU detected.")
        return

    device_count = torch.cuda.device_count()
    
    for i in range(device_count):
        # Get free and total memory in bytes
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        
        # Convert to Gigabytes (GB)
        total_gb = total_mem / (1024 ** 3)
        free_gb = free_mem / (1024 ** 3)
        used_gb = total_gb - free_gb
        
        logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        logging.info(f"Total Memory: {total_gb:.2f} GB")
        logging.info(f"Used Memory:  {used_gb:.2f} GB")
        logging.info(f"Free Memory:  {free_gb:.2f} GB")
        logging.info("-" * 30)

def header_log(text):
    dashes = "----------------------------"
    logging.info(dashes + text + dashes)

def print_state_dict_dtype_counts(state_dict):
    """
    Iterates through a model state dict, counts the number of tensors 
    for each dtype (bf16, fp32, etc.), and prints the summary.
    """
    dtype_counts = Counter()
    total_params = 0

    for key, tensor in state_dict.items():
        # Ensure we are strictly looking at tensors
        if torch.is_tensor(tensor):
            dtype_counts[tensor.dtype] += 1
            total_params += 1
        else:
            # Handle cases where state_dict might contain non-tensor data (rare but possible)
            dtype_counts[type(tensor)] += 1

    logging.info("State Dict Dtype Distribution:")
    for dtype, count in dtype_counts.items():
        logging.info(f"  {dtype}: {count} items ({count/total_params:.1%})")
    
    logging.info(f"  Total items: {total_params}")

def print_model_param_dtypes(model):
    """
    Iterates over a model's parameters to count how many are bf16, fp32, etc.
    """
    dtype_counts = Counter()
    total_params = 0

    for param in model.parameters():
        dtype_counts[param.dtype] += 1
        total_params += 1

    logging.info(f"Model ({type(model).__name__}) Parameter Dtype Distribution:")
    for dtype, count in dtype_counts.items():
        logging.info(f"  {dtype}: {count} parameters ({count/total_params:.1%})")

def check_weight_precision(model, accelerator):
    """
    Checks if FP32 weights are actually just padded BF16 weights.
    """
    print("="*50)
    print("STARTING PRECISION CHECK")
    print("="*50)
    
    # Unwrap the model to handle FSDP/DDP wrappers safely
    unwrapped_model = accelerator.unwrap_model(model)
    
    # Access the specific base model you mentioned
    # usage of getattr handles cases where unwrapping might differ slightly
    base_model = getattr(unwrapped_model.model_adapter, "base_model", None)
    
    if base_model is None:
        print("Could not find model.model_adapter.base_model. Checking root model instead.")
        base_model = unwrapped_model

    total_layers = 0
    bf16_layers = 0
    fp32_layers = 0
    
    # Iterate through parameters
    for name, param in base_model.named_parameters():
        # skip non-float32 params (e.g. if you have some int/bool buffers or actual bf16)
        if param.dtype != torch.float32:
            continue
            
        total_layers += 1
        
        # We only need to check a small slice of the tensor to be sure
        # Flatten and take first 1000 elements to save compute/memory
        flat_param = param.data.flatten()[:1000] 
        
        # 1. Downcast to BF16
        # 2. Upcast back to FP32
        param_roundtrip = flat_param.to(torch.bfloat16).to(torch.float32)
        
        # Check for exact bitwise equality
        # If they are equal, the original FP32 had no info in the lower 16 bits
        if torch.equal(flat_param, param_roundtrip):
            bf16_layers += 1
            # print(f"[EFFECTIVELY BF16] {name}") # Uncomment to see specific layers
        else:
            fp32_layers += 1
            max_diff = (flat_param - param_roundtrip).abs().max().item()
            print(f"[TRUE FP32]        {name} | Max Diff: {max_diff:.10f}")

    print("-" * 30)
    print(f"Total FP32 Layers Checked: {total_layers}")
    print(f"Effectively BF16 Layers:   {bf16_layers}")
    print(f"True FP32 Layers:          {fp32_layers}")
    
    if bf16_layers == total_layers:
        print("\nCONCLUSION: Your model is purely BF16 wrapped in FP32 containers.")
    elif fp32_layers > 0:
        print("\nCONCLUSION: Your model contains real FP32 information.")
    print("="*50)


def main():
    run = None
    wandb_kwargs = {
        "entity": 'taudio',
        "project": 'gpu_mem',
        "name": 'gpu_mem_test_7b',
    }

    run = wandb.init(**wandb_kwargs)
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Unified train + distributed eval for TAudio.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--load-checkpoint', type=str, default=None, help='Path to a checkpoint to load for evaluation only')

    args = parser.parse_args()

    # Initialize config manager
    config_manager = ConfigManager()

    # Load configuration
    config = config_manager.load_config(f"{args.config}")

    model_config = config['model']
    loss_config = config['loss']
    dataset_config = config['dataset']
    task_config = config['task']
    training_config = config['training']
    system_config = config['system']

    task = create_task(task_type=task_config['type'], **task_config.get('kwargs', {}))

    # Build model
    taudio_config = {
        **model_config,
        **loss_config,
        "task": task
    }
    logging.info("About to create model")
    print_gpu_memory()

    logging.info("Creating model")
    model = TAudio(**taudio_config)
    logging.info("Created model")
    print_gpu_memory()
    print_model_param_dtypes(model)

    # random bug where if gradients arent tracked they cause a keyerror
    # for some reason audio_bos_eos_token is also an unused param
    model.model_adapter.base_model.audio_tower.audio_bos_eos_token.requires_grad_(False)
    model.model_adapter.base_model.visual.requires_grad_(False)

    if not loss_config.get('surrogate_loss', False):
        model.linear.requires_grad_(False)

    if not loss_config.get('token_loss', False):
        model.model_adapter.base_model.lm_head.requires_grad_(False)

    model.train()

    accelerator = Accelerator()

    optim = torch.optim.AdamW(model.parameters(), lr=training_config['learning_rate'])

    # Accelerator prepare
    logging.info("Using accelerator to prepare model and ghost optimizer")

    model, optim = accelerator.prepare(model, optim)
    logging.info(model.model_adapter.dtype)

    print_gpu_memory()
    print_model_param_dtypes(model)

    temp_optimizers = accelerator._optimizers
    accelerator._optimizers = []

    # Accelerator load_state
    logging.info("Loading checkpoint state")
    accelerator.load_state(args.load_checkpoint)

    print_gpu_memory()
    print_model_param_dtypes(model)

    # logging.info("Casting underlying model to bfloat16")
    # unwrapped = accelerator.unwrap_model(model)
    # unwrapped.to(torch.bfloat16)

    # print_gpu_memory()
    # print_model_param_dtypes(model)

    # Collecting state dict
    header_log("Collecting model state dict")

    full_state = get_model_state_dict(
        accelerator.unwrap_model(model),
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        cpu_offload=True
    )

    print_state_dict_dtype_counts(full_state)

    print_gpu_memory()

    # Clearing cache
    header_log("Clearing cache")

    import gc; gc.collect()
    torch.cuda.empty_cache()

    print_gpu_memory()

    # Create eval model and load state dict
    header_log("Loading new state dict into eval_model")

    eval_model = TAudio(**taudio_config)
    
    header_log("Checking precision of eval_model before loading state_dict")
    print_model_param_dtypes(eval_model)

    header_log(f"Eval model adapter dtype before loading state dict: {eval_model.model_adapter.dtype}")
    header_log(f"Eval model linear dtype before loading state dict: {eval_model.linear.weight.dtype}")

    eval_model.load_state_dict(full_state, strict=True)

    header_log("Checking precision of eval_model after loading state_dict")
    print_model_param_dtypes(eval_model)

    header_log(f"Eval model adapter dtype after loading state dict: {eval_model.model_adapter.dtype}")
    header_log(f"Eval model linear dtype after loading state dict: {eval_model.linear.weight.dtype}")

    print_gpu_memory()

    # Move model to cpu
    header_log("Moving original model back to cpu")

    header_log(f"Model adapter dtype before sending to GPU: {model.model_adapter.dtype}")
    header_log(f"Model linear dtype before sending to GPU: {model.linear.weight.dtype}")

    model.to('cpu')

    header_log(f"Model adapter dtype after sending to GPU: {model.model_adapter.dtype}")
    header_log(f"Model linear dtype after sending to GPU: {model.linear.weight.dtype}")

    print_gpu_memory()

    # Clear cache
    header_log("Clearing cache")

    import gc; gc.collect()
    torch.cuda.empty_cache()

    print_gpu_memory()

    # Move eval model to GPU
    header_log("Moving eval model to gpu")

    header_log(f"Eval model adapter dtype before sending to GPU: {eval_model.model_adapter.dtype}")
    header_log(f"Eval model linear dtype before sending to GPU: {eval_model.linear.weight.dtype}")

    eval_model.to(accelerator.device)

    header_log("Checking eval_model precision after sending to GPU")
    print_model_param_dtypes(eval_model)

    header_log(f"Eval model adapter dtype after sending to GPU: {eval_model.model_adapter.dtype}")
    header_log(f"Eval model linear dtype after sending to GPU: {eval_model.linear.weight.dtype}")

    print_gpu_memory()

    header_log("Deleting model (not clearing cache)")
    del model
    print_gpu_memory()

    header_log("Clearing cache")
    import gc; gc.collect()
    torch.cuda.empty_cache()

    print_gpu_memory()

    header_log("Deleting state dict")
    del full_state
    print_gpu_memory()

    header_log("Clearing cache")
    import gc; gc.collect()
    torch.cuda.empty_cache()

    print_gpu_memory()
    
    header_log("Performing cast check")

    logging.info("--- Buffer Check BEFORE .to(bf16) ---")
    fp32_bufs = [name for name, buf in eval_model.named_buffers() if buf.dtype == torch.float32]
    logging.info(f"FP32 Buffers count: {len(fp32_bufs)}")
    if len(fp32_bufs) > 0:
        logging.info(f"Examples: {fp32_bufs[:3]}") # Likely 'rotary_emb.inv_freq' etc.

    weight_before = eval_model.linear.weight.detach().clone()
    dtype_before = weight_before.dtype
    
    # 2. Perform the cast
    eval_model.to(torch.bfloat16)
    
    # 3. Snapshot the weight after
    weight_after = eval_model.linear.weight
    dtype_after = weight_after.dtype

    # 4. detailed Comparison
    logging.info(f"Weight Cast Check: {dtype_before} -> {dtype_after}")
    
    if dtype_before == dtype_after:
        logging.info("  Dtypes are identical. Checking for bitwise equality...")
        is_equal = torch.equal(weight_before, weight_after)
        logging.info(f"  Exact Bitwise Equality: {is_equal}")
    else:
        logging.info("  Dtypes changed. Checking numerical difference (quantization noise)...")
        
        # Cast both to float32 to compare numerical values
        diff = (weight_before.float() - weight_after.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        non_zero_diffs = (diff > 0).sum().item()
        total_params = diff.numel()

        logging.info(f"  Max Absolute Difference: {max_diff:.8f}")
        logging.info(f"  Mean Difference:         {mean_diff:.8f}")
        logging.info(f"  Changed Elements:        {non_zero_diffs} / {total_params} ({non_zero_diffs/total_params:.1%})")

        if max_diff > 0:
            logging.info("  CONCLUSION: The weights numerically CHANGED due to precision reduction.")
        else:
            logging.info("  CONCLUSION: The weights are numerically IDENTICAL (no precision lost).")

    logging.info("--- Buffer Check AFTER .to(bf16) ---")
    fp32_bufs_after = [name for name, buf in eval_model.named_buffers() if buf.dtype == torch.float32]
    bf16_bufs_after = [name for name, buf in eval_model.named_buffers() if buf.dtype == torch.bfloat16]
    
    logging.info(f"FP32 Buffers count: {len(fp32_bufs_after)}")
    logging.info(f"BF16 Buffers count: {len(bf16_bufs_after)}")
    
    if len(fp32_bufs) > 0 and len(fp32_bufs_after) == 0:
        logging.info("CONCLUSION: The cast destroyed the precision of your FP32 buffers (likely RoPE).")

    run.finish()

if __name__ == "__main__":
    main()
