import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
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


def limit_dataset(dataset, max_examples: int):
    """Return a subset of the dataset capped at max_examples."""
    if len(dataset) <= max_examples:
        return dataset
    return Subset(dataset, range(max_examples))


def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Unified train + distributed eval for TAudio.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--no-timestamp', action='store_true', help='Don\'t add timestamp to output directory name')
    parser.add_argument('--debug', action='store_true', help='Don\'t log to wandb or experiment directory, and don\'t save model checkpoints')

    parser.add_argument('--eval-min-time', type=float, default=None, help='Minimum time for evaluating on test split')
    parser.add_argument('--eval-max-time', type=float, default=None, help='Maximum time for evaluating on test split')
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

    # Process/world info
    world_size = PartialState().num_processes
    batch_size_per_device = training_config['effective_batch_size'] // max(world_size, 1)
    is_master = PartialState().is_main_process

    torch.cuda.set_device(PartialState().device)
    logging.info(f"World Size: {world_size}")
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
    experiment_dir: Path = Path("tmp_checkpoints")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = relative_path_to_experiment_name(args.config, eval=False)
    project_name = relative_path_to_project_name(args.config, eval=False)

    if not args.debug:
        # Save config to fixed checkpoint directory
        config_manager.save_config(config, experiment_dir)
    logging.info(f"Output directory: {experiment_dir}")
    logging.info(f"Project name: {project_name}")
    logging.info(f"Starting experiment: {experiment_name}")

    # Initialize wandb
    run = None

    # Create task
    task = create_task(task_type=task_config['type'], **task_config.get('kwargs', {}))

    # Build model
    taudio_config = {
        **model_config,
        **loss_config,
        "task": task
    }
    model = TAudio(**taudio_config)

    model.train()

    accelerator = Accelerator()

    resume_state_dir = args.load_checkpoint if args.load_checkpoint and os.path.isdir(args.load_checkpoint) else None

    if resume_state_dir is None and args.load_checkpoint:
        # Evaluation-only path (checkpoint file or non-accelerate dir). For FSDP2 we still need an optimizer.
        dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model, dummy_optimizer = accelerator.prepare(model, dummy_optimizer)
    else:
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

        ds = limit_dataset(ds, 100)
        logging.info(f"Debug mode: limiting training dataset to {len(ds)} examples")

        dataloader = DataLoader(
            ds,
            batch_size=batch_size_per_device,
            drop_last=True,
            pin_memory=True,
            num_workers=8,
            collate_fn=collate_fn,
            shuffle=True
        )

        model.model_adapter.base_model.audio_tower.audio_bos_eos_token.requires_grad_(False)
        model.model_adapter.base_model.visual.requires_grad_(False)

        if not loss_config.get('surrogate_loss', False):
            model.linear.requires_grad_(False)

        optim = torch.optim.AdamW(model.parameters(), lr=training_config['learning_rate'])
        num_optim_steps = len(dataloader) * training_config['epochs']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=num_optim_steps,
            eta_min=training_config['learning_rate'] * training_config['eta_min_scale'],
        )

        model, optim, scheduler, dataloader = accelerator.prepare(model, optim, scheduler, dataloader)
        logging.info(f"Number of optimizer steps: {num_optim_steps}")
        logging.info(f"Dataloader length: {len(dataloader)}")

        if resume_state_dir:
            logging.info(f"Resuming training state from {resume_state_dir}")
            accelerator.load_state(resume_state_dir)

    # Flags for what to evaluate
    eval_token_outputs = bool(loss_config.get('token_loss', False))
    eval_aux_outputs = bool(loss_config.get('surrogate_loss', False))

    if accelerator.is_main_process:
        print("\n" + "="*30)
        print("INSPECTING FROZEN PARAMETERS")
        print("="*30)
        
        # We unwrap the model to ensure we get the original variable names
        # instead of FSDP wrapper names.
        unwrapped_model = accelerator.unwrap_model(model)
        
        frozen_count = 0
        trainable_count = 0
        
        for name, param in unwrapped_model.named_parameters():
            if not param.requires_grad:
                print(f"❄️  FROZEN: {name}  [Shape: {param.shape}]")
                frozen_count += 1
            else:
                trainable_count += 1

        print("-" * 30)
        print(f"Total Frozen Parameters:    {frozen_count}")
        print(f"Total Trainable Parameters: {trainable_count}")
        print("="*30 + "\n")

    # Helper: distributed evaluation
    def get_full_state_dict(state_dir: str | None):
        """
        Helper that returns a full, unsharded model state dict.
        When a checkpoint directory is provided, we load it with a fresh Accelerator
        that tracks only the model so we avoid optimizer/scheduler loading errors.
        """
        if state_dir is not None:

            # load_accelerator = Accelerator()
            # load_model = TAudio(**taudio_config)

            # load_model.model_adapter.base_model.audio_tower.audio_bos_eos_token.requires_grad_(False)
            # load_model.model_adapter.base_model.visual.requires_grad_(False)

            # if not loss_config.get('surrogate_loss', False):
            #     model.linear.requires_grad_(False)

            # dummy_optimizer = torch.optim.AdamW(load_model.parameters(), lr=1e-3)
            # load_model, _ = load_accelerator.prepare(load_model, dummy_optimizer)
            accelerator.load_state(state_dir)
            full_state = get_model_state_dict(
                accelerator.unwrap_model(model),
                options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
            )
            # Best-effort cleanup
            # del load_model
            # del dummy_optimizer
            # del load_accelerator
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

        # If either bound is specified for this eval, explicitly set both,
        # allowing None to clear prior training-time constraints.
        if min_time is not None or max_time is not None:
            task.min_time = min_time
            task.max_time = max_time

        full_state_dict = get_full_state_dict(state_dir)

        eval_model = TAudio(**taudio_config)
        eval_model.load_state_dict(full_state_dict, strict=True)
        eval_model.to(accelerator.device)
        eval_model.eval()

        # Use raw dataset through adapter and shard across processes
        adapter = create_adapter(
            infer_adapter_from_repository(dataset_config['repository']),
            repository=dataset_config['repository'],
            sampling_rate=model.model_adapter.sampling_rate,
            left_padding=dataset_config.get('left_padding', 0),
            key=task.key,
            take_first=dataset_config.get('take_first', None),
        )

        base_ds = adapter.load_split(split_name)
        base_ds = limit_dataset(base_ds, 100)
        logging.info(f"Debug mode: limiting {split_name} split to {len(base_ds)} examples for eval")

        # Shard across processes using Accelerate context manager
        distributed_state = PartialState()
        with distributed_state.split_between_processes(base_ds) as ds_shard:
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

        # Collect keys and reduce sums and counts across processes using a fixed schema
        # so that all ranks execute identical reduce calls in identical order.
        token_metric_keys: List[str] = [
            "token_abs_error_sum",
            "token_correct_5ms",
            "token_correct_10ms",
            "token_correct_20ms",
            "token_correct_40ms",
            "token_correct_50ms",
            "token_correct_80ms",
            "token_correct_100ms",
            "token_correct_200ms",
            "parsing_error",
        ]
        aux_metric_keys: List[str] = [
            # Timestamp-style aux metrics
            "aux_abs_error_sum",
            "aux_correct_5ms",
            "aux_correct_10ms",
            "aux_correct_20ms",
            "aux_correct_40ms",
            "aux_correct_50ms",
            "aux_correct_80ms",
            "aux_correct_100ms",
            "aux_correct_200ms",
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
            # if epoch is not None:
            #     log_payload["train/epoch"] = epoch + 1
            run.log(log_payload)
        
        task.min_time = original_min_time
        task.max_time = original_max_time

        # cleanup
        del eval_model
        del full_state_dict
        del adapter
        del base_ds
        import gc; gc.collect()
        torch.cuda.empty_cache()

        return aggregated

    # Training loop
    epochs = training_config['epochs'] if resume_state_dir is not None else (0 if args.load_checkpoint else training_config['epochs'])

    best_metric = float('-inf') # infinite abs error is worst case
    best_checkpoint_dir = None

    for epoch in range(epochs):
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=not is_master,
        )
        metrics = AverageMetrics()

        for step, batch in enumerate(progress_bar, start=1):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            output = model(**batch)
    

            accelerator.backward(output.loss)
            optim.step()
            optim.zero_grad()
            scheduler.step()

            loss = accelerator.reduce(output.loss, reduction='mean')
            token_loss = accelerator.reduce(output.token_loss, reduction='mean')
            surrogate_loss = accelerator.reduce(output.surrogate_loss, reduction='mean')
            auxiliary_deviation = accelerator.reduce(output.auxiliary_deviation, reduction='mean')

            metrics.update_dict({
                "train/loss": loss.item(),
                "train/token_loss": token_loss.item(),
                "train/surrogate_loss": surrogate_loss.item(),
                "train/auxiliary_deviation": auxiliary_deviation.item(),
            })

            if is_master and not args.debug and run is not None:
                run.log({
                    **metrics.to_dict(),
                    "train/epoch": epoch + 1,
                    "train/step": step + 1,
                    "train/lr": scheduler.get_last_lr()[0],
                })
                metrics.reset()

            progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

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
            if current_metric > best_metric: # 100ms accuracy should be higher
                best_metric = current_metric
                best_checkpoint_dir = checkpoint_dir
                if is_master:
                    logging.info(f"New best model found with {target_metric}: {best_metric}")
        
        accelerator.wait_for_everyone()

    # Final evaluation on test split

    final_checkpoint = best_checkpoint_dir if best_checkpoint_dir else args.load_checkpoint
    if final_checkpoint:
        logging.info(f"Evaluating with checkpoint: {final_checkpoint}")

    test_split = 'test'
    distributed_eval(test_split, prefix="test", epoch=training_config['epochs'] - 1, state_dir=final_checkpoint) # first do evaluation on the constraints imposed during training

    accelerator.wait_for_everyone()

    if args.eval_min_time is not None or args.eval_max_time is not None:
        distributed_eval(test_split, prefix="test_ood", epoch=training_config['epochs'] - 1, min_time=args.eval_min_time, max_time=args.eval_max_time, state_dir=final_checkpoint)
    
    accelerator.wait_for_everyone()

    # Completion line
    if is_master:
        logging.info(f"Training completed. All outputs saved to: {experiment_dir}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()



