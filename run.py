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
    flatten_config,
    relative_path_to_experiment_name,
    relative_path_to_project_name,
)
from utils.metrics import AverageMetrics
from test_7b_loading_single_gpu import print_gpu_memory, print_model_param_dtypes, print_state_dict_dtype_counts
from utils.utils import dist_log
from tqdm.contrib.logging import logging_redirect_tqdm

@logging_redirect_tqdm()
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


    logging.info("Checking dtypes before accelerator.prepare")
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
    
    dist_log(accelerator, "Mem usage before prepare")
    print_gpu_memory()

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
    num_optim_steps = len(dataloader) * training_config['epochs']

    # if we are doing legacy .pt loading we will need to prefill the model before sharding it
    if args.load_checkpoint:
        ckpt_path = Path(args.load_checkpoint)
        
        if ckpt_path.is_file():
            logging.info(f"Loading pt")
            print_model_param_dtypes(model)

            logging.info(f"Loading checkpoint directly from file: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location='cpu')
            
            accelerator.unwrap_model(model).load_state_dict(state_dict)

            logging.info("Loaded pt")
            print_model_param_dtypes(model)
            
    # Note: passing optimizer to prepare is crucial for FSDP/Accelerate to handle sharded states
    model, optim, dataloader = accelerator.prepare(model, optim, dataloader)

    logging.info("Checking dtypes after accelerator.prepare")
    print_model_param_dtypes(model)

    dist_log(accelerator, "Mem usage after prepare")
    print_gpu_memory()

    logging.info(f"Number of optimizer steps: {num_optim_steps}")
    logging.info(f"Dataloader length: {len(dataloader)}")

    if args.eval_only:
        logging.info("Hiding accelerator optimizers for evaluations")
        accelerator._optimizers = []

    start_epoch = 0
    if args.load_checkpoint and not args.eval_only: # we will immediately skip to eval if eval_only which already loads the ckpt
        ckpt_path = Path(args.load_checkpoint)
        
        if ckpt_path.is_file():
            logging.info(".pt detected, skipping second load")
                
        else:
            # Load from Accelerate directory structure
            accelerator.load_state(args.load_checkpoint)

            logging.info("Checking dtypes after accelerator.load_state")
            print_model_param_dtypes(model)

            dist_log(accelerator, "Mem after accelerator.load_state")
            print_gpu_memory()
            
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
        """
        Helper that returns a full, unsharded model state dict.
        """
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
        logging.info("Clearing cache before eval")
        torch.cuda.empty_cache()
        import gc; gc.collect()

        original_min_time = task.min_time
        original_max_time = task.max_time

        # If either bound is specified for this eval, explicitly set both,
        # allowing None to clear prior training-time constraints.
        if min_time is not None or max_time is not None:
            task.min_time = min_time
            task.max_time = max_time

        dist_log(accelerator, "Mem before collecting state dict")
        print_gpu_memory()
        logging.info("Collecting state dict")
        full_state_dict = get_full_state_dict(state_dir)
        dist_log(accelerator, "Mem after collecting state dict")
        print_gpu_memory()

        print_state_dict_dtype_counts(full_state_dict)

        eval_model = TAudio(**taudio_config)

        dist_log(accelerator, "Mem after creating eval_model")
        print_gpu_memory()

        logging.info("Checking eval model parameters before loading state dict")
        print_model_param_dtypes(eval_model)

        eval_model.load_state_dict(full_state_dict, strict=True)
        logging.info("State dict loaded. Deleting copy to free RAM...")

        logging.info("State dict loaded. Deleting copy to free RAM...")
        del full_state_dict
        import gc; gc.collect()
        torch.cuda.empty_cache()

        dist_log(accelerator, "Mem after loading state dict into eval_model and deleting full_state_dict")
        print_gpu_memory()

        if args.eval_only:
            # note that this likely destroys the model for any later evaluations (joint, ood)
            model.to('cpu')
            import gc; gc.collect()
            torch.cuda.empty_cache()

            dist_log(accelerator, "Mem after moving base model to cpu")
            print_gpu_memory()

        eval_model.to(accelerator.device)
        eval_model.eval()

        dist_log(accelerator, "Mem after moving eval model to GPU")
        print_gpu_memory()

        logging.info("Checking eval model parameters after loading state dict and sending to GPU")
        print_model_param_dtypes(eval_model)

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
        base_ds = task.select_indices(base_ds, adapter, split_name)

        # Shard across processes using Accelerate context manager
        distributed_state = PartialState()
        with distributed_state.split_between_processes(base_ds) as ds_shard:
            print(f"Base dataset length: {len(base_ds)}")
            for example in ds_shard:
                print(example)
                break

        local_metrics = AverageMetrics()
        with distributed_state.split_between_processes(base_ds) as ds_shard:
            ds_shard_pbar = tqdm(ds_shard, desc=f"Rank {accelerator.process_index}")

            # aux_batch = []
            # EVAL_AUX_BATCH_SIZE = 8

            for example in ds_shard_pbar:
                if task.skip_example(example, adapter):
                    continue
                if eval_token_outputs:
                    token_metrics = task.evaluate_tokens(
                        example=example,
                        ds_adapter=adapter,
                        model=eval_model,
                    )
                    if token_metrics is not None:
                        local_metrics.update_dict(token_metrics)

                if eval_aux_outputs:
                    # OLD
                    aux_metrics = task.evaluate_auxiliary_outputs(
                        example=example,
                        ds_adapter=adapter,
                        model=eval_model,
                        error_bound=0.1,
                    )
                    if aux_metrics is not None:
                        local_metrics.update_dict(aux_metrics)
            #         aux_batch.append(example)

            #         if len(aux_batch) >= EVAL_AUX_BATCH_SIZE:
            #             aux_metrics_list = task.evaluate_auxiliary_outputs_batched(
            #                 examples=aux_batch,
            #                 ds_adapter=adapter,
            #                 model=eval_model,
            #             )
            #             for aux_metrics in aux_metrics_list:
            #                 if aux_metrics is not None:
            #                     local_metrics.update_dict(aux_metrics)
            #             aux_batch = [] # Reset batch
            
            # if eval_aux_outputs and aux_batch:
            #     aux_metrics_list = task.evaluate_auxiliary_outputs_batched(
            #         examples=aux_batch,
            #         ds_adapter=adapter,
            #         model=eval_model,
            #     )
            #     for aux_metrics in aux_metrics_list:
            #         if aux_metrics is not None:
            #             local_metrics.update_dict(aux_metrics)


        to_reduce = {}
        for key in sorted(local_metrics._sum.keys()):
            to_reduce[f"{key}_sum"] = torch.tensor(local_metrics.get_sum(key), device=accelerator.device)
            to_reduce[f"{key}_cnt"] = torch.tensor(local_metrics.get_count(key), device=accelerator.device)

        reduced = accelerator.reduce(to_reduce, reduction='sum')

        # 3. Calculate averages
        aggregated = {
            f"{prefix}/{key}": (reduced[f"{key}_sum"] / torch.clamp_min(reduced[f"{key}_cnt"], 1.0)).item()
            for key in local_metrics._sum.keys()
        }

        if is_master and not args.debug and run is not None:
            log_payload = dict(aggregated)
            # if epoch is not None:
            #     log_payload["train/epoch"] = epoch + 1
            run.log(log_payload)
        
        task.min_time = original_min_time
        task.max_time = original_max_time

        # cleanup
        del eval_model
        del adapter
        del base_ds
        import gc; gc.collect()
        torch.cuda.empty_cache()

        return aggregated

    # Training loop
    epochs = 0 if args.eval_only else training_config['epochs']

    best_metric = float('-inf') # negative inf accuracy is the worst case
    best_checkpoint_dir = args.load_checkpoint if args.eval_only else None

    # this is only for joint training stuff
    best_token_metric = float('-inf') # negative inf accuracy is the worst case
    best_poisson_metric = float('-inf') # negative inf accuracy is the worst case

    best_token_checkpoint_dir = None
    best_poisson_checkpoint_dir = None

    for epoch in range(start_epoch, epochs):
        progress_bar = tqdm(
            dataloader,
            disable=not is_master,
        )
        metrics = AverageMetrics()

        for step, batch in enumerate(progress_bar, start=1):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            output = model(**batch)

            accelerator.backward(output.loss)
            optim.step()
            optim.zero_grad()

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
                })
                metrics.reset()

            progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Rank: {accelerator.process_index}")

        logging.info(f"Epoch {epoch + 1} completed.")

        # Save checkpoint
        accelerator.wait_for_everyone()
        if not args.debug:
            checkpoint_dir = experiment_dir / f"checkpoint_epoch{epoch+1}"
            if is_master:
                logging.info(f"Saving accelerator state to {checkpoint_dir}")
            accelerator.save_state(checkpoint_dir)

        accelerator.wait_for_everyone()

        if not args.debug:
            checkpoint_dir = experiment_dir / f"checkpoint_epoch{epoch+1}"
            dev_split = dataset_config.get('dev_split', 'dev')
            metrics = distributed_eval(dev_split, prefix="dev", epoch=epoch, state_dir=None)
            
            target_metric = "dev/token_correct_40ms"
            if not eval_token_outputs and eval_aux_outputs:
                if loss_config['poisson_loss']:
                    target_metric = "dev/smooth_40ms_boxcar_fixed_posterior_mode/aux_correct_40ms"
                else:
                    target_metric = 'dev/smooth_40ms_boxcar/aux_correct_40ms'
                
            current_metric = metrics.get(target_metric, -1.0)
            logging.info(f"Current model achieved {target_metric}: {current_metric}")
            if current_metric > best_metric: # accuracy should be higher
                best_metric = current_metric
                best_checkpoint_dir = checkpoint_dir
                if is_master:
                    logging.info(f"New best model found with {target_metric}: {best_metric}")

            # joint training check
            if loss_config['token_loss'] and loss_config['poisson_loss']:
                token_metric = metrics.get('dev/token_correct_40ms')
                poisson_metric = metrics.get('dev/smooth_40ms_boxcar_fixed_posterior_mode/aux_correct_40ms')

                if token_metric > best_token_metric:
                    best_token_metric = token_metric
                    best_token_checkpoint_dir = checkpoint_dir
                    logging.info(f"New best token model found with {token_metric} at {best_token_checkpoint_dir}")

                if poisson_metric > best_poisson_metric:
                    best_poisson_metric = poisson_metric
                    best_poisson_checkpoint_dir = checkpoint_dir
                    logging.info(f"New best poisson model found with {poisson_metric} at {best_poisson_checkpoint_dir}")

        
        accelerator.wait_for_everyone()

    # Final evaluation on test split

    final_checkpoint = best_checkpoint_dir if best_checkpoint_dir else args.load_checkpoint
    if final_checkpoint:
        logging.info(f"Evaluating with checkpoint: {final_checkpoint}")

    if args.dev:
        split = 'dev'
    else:
        split = 'test'
    
    logging.info(f"Evaluating final split on split {split}")

    # super hacky but basically we don't care about loading checkpoints for joint training
    # also don't care about doing ood evals for joint training
    if loss_config['token_loss'] and loss_config['poisson_loss']:
        if best_token_checkpoint_dir == best_poisson_checkpoint_dir:
            distributed_eval(split, prefix=f'token+poisson-{split}', epoch=training_config['epochs'] - 1, state_dir=best_token_checkpoint_dir) # first do evaluation on the constraints imposed during training
        else:
            distributed_eval(split, prefix=f'token-{split}', epoch=training_config['epochs'] - 1, state_dir=best_token_checkpoint_dir) # first do evaluation on the constraints imposed during training
            distributed_eval(split, prefix=f'poisson-{split}', epoch=training_config['epochs'] - 1, state_dir=best_poisson_checkpoint_dir) # first do evaluation on the constraints imposed during training

    else:
        distributed_eval(split, prefix=split, epoch=training_config['epochs'] - 1, state_dir=final_checkpoint) # first do evaluation on the constraints imposed during training

    accelerator.wait_for_everyone()

    if args.eval_min_time is not None or args.eval_max_time is not None:
        distributed_eval(split, prefix=split+"_ood", epoch=training_config['epochs'] - 1, min_time=args.eval_min_time, max_time=args.eval_max_time, state_dir=final_checkpoint)
    
    accelerator.wait_for_everyone()

    # Completion line
    if is_master:
        logging.info(f"Training completed. All outputs saved to: {experiment_dir}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()