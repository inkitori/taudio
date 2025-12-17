import argparse
import logging
import os
import random
import re
import math
import contextlib
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
    flatten_config,
    relative_path_to_experiment_name,
    relative_path_to_project_name,
)
from utils.metrics import AverageMetrics
from utils.utils import dist_log


def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Unified train + distributed eval for TAudio.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--no-timestamp', action='store_true', help='Don\'t add timestamp to output directory name')
    parser.add_argument('--debug', action='store_true', help='Don\'t log to wandb or experiment directory, and don\'t save model checkpoints')

    parser.add_argument('--eval-min-time', type=float, default=None, help='Minimum time for evaluating on test split')
    parser.add_argument('--eval-max-time', type=float, default=None, help='Maximum time for evaluating on test split')
    parser.add_argument('--load-checkpoint', type=str, default=None, help='Path to a checkpoint to load for evaluation only')
    parser.add_argument('--eval-only', action='store_true', help='Don\'t run training')
    parser.add_argument('--run-id', type=str, default=None, help='wandb run id')
    parser.add_argument('--dev', action='store_true', help='Perform final evaluation on dev')

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

    # Process/world info
    world_size = PartialState().num_processes
    is_master = PartialState().is_main_process
    
    # --------------------------------------------------------------------------
    # Batch Size & Gradient Accumulation Calculation
    # --------------------------------------------------------------------------
    effective_batch_size = training_config['effective_batch_size']
    
    # Calculate per-device batch size
    # Formula: Effective = Per_Device * World_Size * Accum_Steps
    batch_size_per_device = 1
    grad_accum_steps = effective_batch_size // (batch_size_per_device * world_size)
    
    if batch_size_per_device < 1:
        raise ValueError("Calculated batch size per device was 0")

    torch.cuda.set_device(PartialState().device)
    logging.info(f"World Size: {world_size}")
    logging.info(f"Effective Batch Size: {effective_batch_size}")
    logging.info(f"Gradient Accumulation Steps: {grad_accum_steps}")
    logging.info(f"Batch size per device: {batch_size_per_device}")
    logging.info(f"Is master: {is_master}")
    logging.info(f"Using device: {torch.cuda.current_device()}")

    # Set random seed
    random.seed(system_config['seed'])
    torch.cuda.manual_seed(system_config['seed'])
    torch.manual_seed(system_config['seed'])
    np.random.seed(system_config['seed'])
    set_seed(system_config['seed'])

    # Experiment, project
    experiment_dir: Path = None  # type: ignore
    experiment_name = relative_path_to_experiment_name(args.config, eval=False)
    project_name = relative_path_to_project_name(args.config, eval=False)

    # If resuming from a checkpoint, keep using its parent experiment directory.
    if args.load_checkpoint is not None:
        experiment_dir = Path(args.load_checkpoint).resolve().parent
        logging.info(f"Using experiment directory from checkpoint: {experiment_dir}")

    if experiment_dir is None and not args.debug and not args.eval_only:
        # Create experiment directory and save config
        experiment_dir = config_manager.create_experiment_dir(
            args.config,
            timestamp=not args.no_timestamp
        )
        config_manager.save_config(config, experiment_dir)
    logging.info(f"Output directory: {experiment_dir}")
    logging.info(f"Project name: {project_name}")
    logging.info(f"Starting experiment: {experiment_name}")

    # Initialize wandb
    run = None
    if not args.debug and is_master:
        flattened_config = flatten_config(config)
        wandb_kwargs = {
            "entity": config['wandb']['entity'],
            "project": project_name,
            "name": experiment_name,
            "config": flattened_config,
        }

        # When a run id is provided, attempt to resume that run instead of creating a new one.
        if args.run_id is not None:
            wandb_kwargs["id"] = args.run_id
            wandb_kwargs["resume"] = "allow"
            logging.info(f"Resuming wandb run with id={args.run_id}")

        run = wandb.init(**wandb_kwargs)

    # Create task
    task = create_task(task_type=task_config['type'], **task_config.get('kwargs', {}))

    # Build model
    taudio_config = {
        **model_config,
        **loss_config,
        "task": task
    }
    model = TAudio(**taudio_config)

    # random bug where if gradients arent tracked they cause a keyerror
    # for some reason audio_bos_eos_token is also an unused param
    model.model_adapter.base_model.audio_tower.audio_bos_eos_token.requires_grad_(False)
    model.model_adapter.base_model.visual.requires_grad_(False)

    if not loss_config.get('surrogate_loss', False):
        model.linear.requires_grad_(False)

    if not loss_config.get('token_loss', False):
        model.model_adapter.base_model.lm_head.requires_grad_(False)

    model.train()

    # Initialize Accelerator with specific gradient accumulation steps
    # This helps internal state management, though we do manual accumulation in the loop.
    accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps)

    # Build training dataset/dataloader
    ds, ds_adapter = get_ds(
        model_adapter=model.model_adapter,
        repository=dataset_config['repository'],
        split=dataset_config['split'],
        task=task,
        take_first=dataset_config.get('take_first', None),
        left_padding=dataset_config.get('left_padding', 0),
    )

    accelerator.wait_for_everyone()

    dataloader = DataLoader(
        ds,
        batch_size=batch_size_per_device,
        drop_last=True,
        pin_memory=True,
        num_workers=8,
        collate_fn=collate_fn,
        shuffle=True
    )

    optim = torch.optim.AdamW(model.parameters(), lr=training_config['learning_rate'])
    
    

    # if we are doing legacy .pt loading we will need to prefill the model before sharding it
    if args.load_checkpoint:
        ckpt_path = Path(args.load_checkpoint)
        
        if ckpt_path.is_file():
            logging.info(f"Loading checkpoint directly from file: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location='cpu')
            accelerator.unwrap_model(model).load_state_dict(state_dict)

    # Note: passing optimizer to prepare is crucial for FSDP/Accelerate to handle sharded states
    model, optim, dataloader = accelerator.prepare(model, optim, dataloader)
    
    # Calculate total update steps
    dist_log(accelerator, f"Dataloader Length: {len(dataloader)}")
    
    # preventing deadlocks from gpu stopping early
    local_len = torch.tensor(len(dataloader), device=accelerator.device)
    min_len = accelerator.reduce(local_len, reduction="min").item()
    num_update_steps_per_epoch = int(min_len) // grad_accum_steps
    dist_log(accelerator, f"Adjusted update steps per epoch (deadlock safe): {num_update_steps_per_epoch}")

    num_optim_steps = num_update_steps_per_epoch * training_config['epochs']
    
    logging.info(f"Total batches per epoch: {len(dataloader)}")
    logging.info(f"Optimizer update steps per epoch: {num_update_steps_per_epoch}")
    logging.info(f"Total training steps: {num_optim_steps}")

    start_epoch = 0
    if args.load_checkpoint:
        ckpt_path = Path(args.load_checkpoint)
        
        if ckpt_path.is_file():
            logging.info(".pt detected, skipping second load")
        else:
            # Load from Accelerate directory structure
            accelerator.load_state(args.load_checkpoint)
            
            # Infer start epoch from directory name
            ckpt_name = Path(args.load_checkpoint).name
            epoch_match = re.findall(r'\d+', ckpt_name)
            if epoch_match:
                start_epoch = int(epoch_match[-1])
                logging.info(f"Resuming training from epoch {start_epoch}")
            else:
                logging.warning(f"Could not infer epoch from checkpoint path: {args.load_checkpoint}")

    # Flags for what to evaluate
    eval_token_outputs = bool(loss_config.get('token_loss', False))
    eval_aux_outputs = bool(loss_config.get('surrogate_loss', False))

    # Helper: distributed evaluation
    def get_full_state_dict(state_dir: str | None):
        if state_dir is not None:
            path = Path(state_dir)
            if path.is_file():
                # Direct load from .pt file
                return torch.load(path, map_location='cpu')
            else:
                # Load via Accelerator (directory)
                accelerator.load_state(state_dir)
                full_state = get_model_state_dict(
                    accelerator.unwrap_model(model),
                    options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
                )
                import gc; gc.collect()
                torch.cuda.empty_cache()
                return full_state

        unwrapped_model = accelerator.unwrap_model(model)
        return get_model_state_dict(
            unwrapped_model,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        )

    def distributed_eval(split_name: str, prefix: str, epoch: int = None, min_time: float = None, max_time: float = None, state_dir: str = None) -> Dict[str, float]:
        original_min_time = task.min_time
        original_max_time = task.max_time

        if min_time is not None or max_time is not None:
            task.min_time = min_time
            task.max_time = max_time

        full_state_dict = get_full_state_dict(state_dir)

        eval_model = TAudio(**taudio_config)
        eval_model.load_state_dict(full_state_dict, strict=True)
        eval_model.to(accelerator.device)
        eval_model.eval()

        adapter = create_adapter(
            infer_adapter_from_repository(dataset_config['repository']),
            repository=dataset_config['repository'],
            sampling_rate=model.model_adapter.sampling_rate,
            left_padding=dataset_config.get('left_padding', 0),
            key=task.key,
            take_first=dataset_config.get('take_first', None),
        )

        base_ds = adapter.load_split(split_name)
        
        # Simple debug print
        distributed_state = PartialState()
        with distributed_state.split_between_processes(base_ds) as ds_shard:
            if is_master:
                print(f"Base dataset length: {len(base_ds)}")
                for example in ds_shard:
                    print(example)
                    break

        local_metrics = AverageMetrics()
        with distributed_state.split_between_processes(base_ds) as ds_shard:
            for example in ds_shard:
                if task.skip_example(example, adapter):
                    continue
                if eval_token_outputs:
                    token_metrics = task.evaluate_tokens(
                        example=example,
                        ds_adapter=adapter,
                        model=eval_model,
                        error_bound=0.1,
                    )
                    if token_metrics is not None:
                        local_metrics.update_dict(token_metrics)
                if eval_aux_outputs:
                    aux_metrics = task.evaluate_auxiliary_outputs(
                        example=example,
                        ds_adapter=adapter,
                        model=eval_model,
                        error_bound=0.1,
                    )
                    if aux_metrics is not None:
                        local_metrics.update_dict(aux_metrics)

        token_metric_keys: List[str] = [
            "token_abs_error_sum", "token_correct_5ms", "token_correct_10ms",
            "token_correct_20ms", "token_correct_40ms", "token_correct_50ms",
            "token_correct_80ms", "token_correct_100ms", "token_correct_200ms",
            "parsing_error",
        ]
        aux_metric_keys: List[str] = [
            "aux_abs_error_sum", "aux_correct_5ms", "aux_correct_10ms",
            "aux_correct_20ms", "aux_correct_40ms", "aux_correct_50ms",
            "aux_correct_80ms", "aux_correct_100ms", "aux_correct_200ms",
        ]

        metric_keys: List[str] = []
        if eval_token_outputs:
            metric_keys.extend(token_metric_keys)
        if eval_aux_outputs:
            metric_keys.extend(aux_metric_keys)

        aggregated: Dict[str, float] = {}
        device = accelerator.device
        for key in metric_keys:
            local_sum = torch.tensor(local_metrics.get_sum(key), device=device, dtype=torch.float32)
            local_cnt = torch.tensor(local_metrics.get_count(key), device=device, dtype=torch.float32)
            global_sum = accelerator.reduce(local_sum, reduction='sum')
            global_cnt = accelerator.reduce(local_cnt, reduction='sum')
            avg = (global_sum / torch.clamp_min(global_cnt, 1.0)).item()
            aggregated[f"{prefix}/{key}"] = avg

        if is_master and not args.debug and run is not None:
            log_payload = dict(aggregated)
            run.log(log_payload)
        
        task.min_time = original_min_time
        task.max_time = original_max_time

        del eval_model
        del full_state_dict
        del adapter
        del base_ds
        import gc; gc.collect()
        torch.cuda.empty_cache()

        return aggregated

    # Training loop
    epochs = 0 if args.eval_only else training_config['epochs']
    best_metric = float('-inf') 
    best_checkpoint_dir = args.load_checkpoint if args.eval_only else None

    for epoch in range(start_epoch, epochs):
        model.train()
        
        # Create an iterator to manually fetch batches for gradient accumulation
        data_iterator = iter(dataloader)
        
        progress_bar = tqdm(
            range(num_update_steps_per_epoch),
            desc=f"Epoch {epoch + 1}",
            disable=not is_master,
        )
        
        metrics = AverageMetrics()

        for update_step in progress_bar:
            # ------------------------------------------------------------------
            # 1. Fetch Batches for this Gradient Accumulation Step
            # ------------------------------------------------------------------
            accum_batches = []
            for _ in range(grad_accum_steps):
                try:
                    accum_batches.append(next(data_iterator))
                except StopIteration:
                    break
            
            if not accum_batches:
                break
            
            # ------------------------------------------------------------------
            # 2. Pre-Calculate Totals for Variable-Size Scaling
            # ------------------------------------------------------------------
            local_step_tokens = 0.0

            for batch in accum_batches:
                # Count valid tokens for Token Loss scaling (CrossEntropy)
                local_step_tokens += (batch['labels'] != -100).sum().item()

            # Convert to tensor for reduction
            # [token_count, sample_count]
            counts_tensor = torch.tensor([local_step_tokens], device=accelerator.device)
            
            # Reduce across all GPUs to get Global Totals for this step
            global_counts = accelerator.reduce(counts_tensor, reduction='sum')
            global_total_tokens = global_counts[0].item()
            
            # ------------------------------------------------------------------
            # 3. Forward & Backward Pass with Mixed Scaling
            # ------------------------------------------------------------------
            # Track logging loss separately

            losses = []
            for i, batch in enumerate(accum_batches):
                # Ensure on device
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                
                # Logic: Only sync gradients on the LAST micro-batch of the accumulation step
                # (Standard Accelerate/DDP optimization)
                ctx = contextlib.nullcontext

                with ctx():
                    output = model(**batch)
                    
                    # --- A. Calculate Local Counts for this specific micro-batch ---
                    current_batch_tokens = (batch['labels'] != -100).sum().item() if 'labels' in batch else 0.0

                    # --- B. Apply Scaling Factors ---
                    # 1. Un-average the loss (multiply by local count)
                    # 2. Re-average over global count (divide by global total)
                    # 3. Multiply by world_size (cancel DDP's implicit division)

                    # Factor for Token Loss (Frame-weighted)
                    token_scale = (current_batch_tokens / global_total_tokens) * world_size * grad_accum_steps
                    
                    # Factor for Surrogate/Aux Loss (Sample-weighted)

                    # Reconstruct total loss
                    # Note: output.token_loss and output.surrogate_loss are local means
                    
                    total_scaled_loss = 0.0
                    
                    if hasattr(output, 'token_loss'):
                        total_scaled_loss += output.token_loss * token_scale
                        
                    if hasattr(output, 'surrogate_loss'):
                        total_scaled_loss += output.surrogate_loss 
                        
                    # If auxiliary_deviation acts as a loss component (unlikely based on name, but safe to check)
                    # Assuming it's just a metric for now based on previous discussions. 
                    # If there are other losses in 'output', add them here with sample_scale.

                        
                    # --- C. Backward ---
                    accelerator.backward(total_scaled_loss)
                    losses.append(total_scaled_loss.detach())
                    
                    # --- D. Update Metrics for Logging ---
                    with torch.no_grad():
                        # We accumulate the raw local means for the metrics dictionary
                        metrics.update_dict({
                            "train/token_loss": output.token_loss.item(),
                            "train/surrogate_loss": output.surrogate_loss.item(),
                            "train/auxiliary_deviation": output.auxiliary_deviation.item(),
                        })
                        

            # ------------------------------------------------------------------
            # 4. Optimization Step
            # ------------------------------------------------------------------
            optim.step()
            optim.zero_grad()

            # ------------------------------------------------------------------
            # 5. Logging
            # ------------------------------------------------------------------

            losses = accelerator.gather(sum(losses)).sum().item() / (
                accelerator.num_processes * grad_accum_steps
            )

            metrics.update_dict({"train/loss": losses})

            if is_master and not args.debug and run is not None:
                run.log({
                    **metrics.to_dict(),
                    "train/epoch": epoch + 1,
                    "train/step": (epoch * num_update_steps_per_epoch) + update_step + 1,
                    # Optional: monitor utilization
                    "train/global_tokens": global_total_tokens,
                })
                metrics.reset()

            progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {losses:.4f}")

        logging.info(f"Epoch {epoch + 1} completed.")

        # Save checkpoint
        accelerator.wait_for_everyone()
        if not args.debug:
            checkpoint_dir = experiment_dir / f"checkpoint_epoch{epoch+1}"
            if is_master:
                logging.info(f"Saving accelerator state to {checkpoint_dir}")
            accelerator.save_state(checkpoint_dir)

        accelerator.wait_for_everyone()

        # Per-epoch distributed eval on dev split
        if not args.debug:
            checkpoint_dir = experiment_dir / f"checkpoint_epoch{epoch+1}"
            dev_split = dataset_config.get('dev_split', 'dev')
            metrics = distributed_eval(dev_split, prefix="dev", epoch=epoch, state_dir=None)
            
            target_metric = "dev/token_correct_100ms"
            if not eval_token_outputs and eval_aux_outputs:
                target_metric = "dev/aux_correct_100ms"
                
            current_metric = metrics.get(target_metric, -1.0)
            logging.info(f"Current model achieved {target_metric}: {current_metric}")
            if current_metric > best_metric: 
                best_metric = current_metric
                best_checkpoint_dir = checkpoint_dir
                if is_master:
                    logging.info(f"New best model found with {target_metric}: {best_metric}")
        
        accelerator.wait_for_everyone()

    # Final evaluation
    final_checkpoint = best_checkpoint_dir if best_checkpoint_dir else args.load_checkpoint
    if final_checkpoint:
        logging.info(f"Evaluating with checkpoint: {final_checkpoint}")

    if args.dev:
        split = 'dev'
    else:
        split = 'test'
    
    logging.info(f"Evaluating final split on split {split}")

    distributed_eval(split, prefix=split, epoch=training_config['epochs'] - 1, state_dir=final_checkpoint)

    accelerator.wait_for_everyone()

    if args.eval_min_time is not None or args.eval_max_time is not None:
        distributed_eval(split, prefix=split+"_ood", epoch=training_config['epochs'] - 1, min_time=args.eval_min_time, max_time=args.eval_max_time, state_dir=final_checkpoint)
    
    accelerator.wait_for_everyone()

    if is_master:
        logging.info(f"Training completed. All outputs saved to: {experiment_dir}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()