import argparse
import logging
import random
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


def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Unified train + distributed eval for TAudio.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--no-timestamp', action='store_true', help='Don\'t add timestamp to output directory name')
    parser.add_argument('--debug', action='store_true', help='Don\'t log to wandb or experiment directory, and don\'t save model checkpoints')
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
    experiment_dir: Path = None  # type: ignore
    experiment_name = relative_path_to_experiment_name(args.config, eval=False)
    project_name = relative_path_to_project_name(args.config, eval=False)

    if not args.debug:
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
        run = wandb.init(
            entity=config['wandb']['entity'],
            project=project_name,
            name=experiment_name,
            config=flattened_config,
        )

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
    num_optim_steps = len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=num_optim_steps,
        eta_min=training_config['learning_rate'] * training_config['eta_min_scale'],
    )

    model, optim, scheduler, dataloader = accelerator.prepare(model, optim, scheduler, dataloader)
    logging.info(f"Number of optimizer steps: {num_optim_steps}")
    logging.info(f"Dataloader length: {len(dataloader)}")

    # Flags for what to evaluate
    eval_token_outputs = bool(loss_config.get('token_loss', False))
    eval_aux_outputs = bool(loss_config.get('surrogate_loss', False))

    # Helper: distributed evaluation
    def distributed_eval(split_name: str, prefix: str, epoch: int = None) -> Dict[str, float]:
        # Build a plain, unwrapped eval model to avoid DTensor/Tensor mixing during generation
        # Get a consolidated state dict from the wrapped model
        unwrapped_model = accelerator.unwrap_model(model)

        full_state_dict = get_model_state_dict(unwrapped_model, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))

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
        )

        base_ds = adapter.load_split(split_name)

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
            if epoch is not None:
                log_payload["train/epoch"] = epoch + 1
            run.log(log_payload)

        return aggregated

    # Training loop
    for epoch in range(training_config['epochs']):
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
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = accelerator.get_state_dict(unwrapped_model)
        if not args.debug and is_master:
            checkpoint_path = experiment_dir / f"model_epoch{epoch+1}.pt"
            logging.info(f"Saving model checkpoint to {checkpoint_path}")
            torch.save(state_dict, checkpoint_path)

        # accelerator.wait_for_everyone()

        # Per-epoch distributed eval on dev split
        # dev_split = dataset_config.get('dev_split', 'dev')
        # distributed_eval(dev_split, prefix="dev", epoch=epoch)

    # Final evaluation on test split
    test_split = dataset_config.get('test_split', 'test')
    distributed_eval(test_split, prefix="test", epoch=training_config['epochs'] - 1)

    # Completion line
    if is_master:
        logging.info(f"Training completed. All outputs saved to: {experiment_dir}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()



