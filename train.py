import argparse
import logging
import random
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, ModuleWrapPolicy
from tqdm.auto import tqdm
import functools
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoderLayer, Qwen2_5OmniDecoderLayer, Qwen2_5OmniAudioEncoder, Qwen2_5OmniThinkerTextModel
import bitsandbytes as bnb
import wandb
from contextlib import nullcontext
from transformers import set_seed
import numpy as np

from dataset.dataset import get_ds, collate_fn
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
    dist.init_process_group(
        backend='nccl',
        timeout=timedelta(hours=1), # an hour
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    is_master = rank == 0

    logging.getLogger().setLevel(logging.INFO)

    logging.info(f"Local rank: {local_rank}")
    logging.info(f"Rank: {rank}")
    logging.info(f"World size: {world_size}")
    logging.info(f"Is master: {is_master}")

    parser = argparse.ArgumentParser(description="Train TAudio model.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config file')
    parser.add_argument('--no-timestamp', action='store_true',
                        help='Don\'t add timestamp to output directory name')
    parser.add_argument('--debug', action='store_true',
                        help='Don\'t log to wandb or experiment directory, and don\'t save model checkpoints')
    args = parser.parse_args()
    
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

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

    # Set random seed
    random.seed(system_config['seed'])
    torch.cuda.manual_seed(system_config['seed'])
    torch.manual_seed(system_config['seed'])
    np.random.seed(system_config['seed'])
    set_seed(system_config['seed'])    

    experiment_dir = None
    experiment_name = relative_path_to_experiment_name(args.config, eval=False)
    project_name = relative_path_to_project_name(args.config, eval=False)

    if not args.debug:
        # Create experiment directory
        experiment_dir = config_manager.create_experiment_dir(
            args.config,
            timestamp=not args.no_timestamp
        )

        # Save config to experiment directory
        config_manager.save_config(config, experiment_dir)


    logging.info(f"Output directory: {experiment_dir}")
    logging.info(f"Project name: {project_name}")
    logging.info(f"Starting experiment: {experiment_name}")
    logging.info(f"Config: {config}")

    batch_size = world_size # 1 batch per device
    gradient_accumulation_steps = training_config['effective_batch_size'] // batch_size

    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    # Create task
    task = create_task(task_type=task_config['type'], **task_config.get('kwargs', {}))

    taudio_config = {
        **model_config,
        **loss_config,
        "task": task
    }

    # Create model
    model = TAudio(**taudio_config)

    transformer_layer_cls = {
        Qwen2_5OmniAudioEncoderLayer,
        Qwen2_5OmniDecoderLayer,
        # Qwen2_5OmniAudioEncoder,
        # Qwen2_5OmniThinkerTextModel,
    }
    
    # auto_wrap_policy = functools.partial(
    #     transformer_auto_wrap_policy,
    #     transformer_layer_cls=transformer_layer_cls,
    # )

    auto_wrap_policy = ModuleWrapPolicy(
        module_classes=transformer_layer_cls
    )
    
    model = FSDP(model, auto_wrap_policy=auto_wrap_policy, device_id=torch.cuda.current_device())
    model.train()

    ds = get_ds(
        model_adapter=model.adapter,
        repository=dataset_config['repository'],
        split=dataset_config['split'],
        task=task,
        take_first=dataset_config.get('take_first', None),
    )

    dataloader = DataLoader(
        ds, 
        batch_size=1, 
        collate_fn=collate_fn,
    )

    dist.barrier() 

    optim = torch.optim.AdamW(model.parameters(), lr=training_config['learning_rate'])

    if not args.debug and is_master:
        flattened_config = flatten_config(config)

        run = wandb.init(
            entity=config['wandb']['entity'],
            project=project_name,
            name=experiment_name,
            config=flattened_config,
        )

    for epoch in range(training_config['epochs']):
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch + 1}",
            disable=not is_master,
            )

        metrics = AverageMetrics()

        for step, batch in enumerate(progress_bar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            context = model.no_sync() if step % gradient_accumulation_steps != 0 else nullcontext()

            with context:
                output = model(**batch)
                
                scaled_loss = output.loss / gradient_accumulation_steps
                    
                scaled_loss.backward()

            dist.all_reduce(output.loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(output.token_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(output.surrogate_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(output.auxiliary_deviation, op=dist.ReduceOp.AVG)

            metrics.update_dict({
                "loss": output.loss.item(),
                "token_loss": output.token_loss.item(),
                "surrogate_loss": output.surrogate_loss.item(),
                "auxiliary_deviation": output.auxiliary_deviation.item(),
            })

            if step % gradient_accumulation_steps == 0:
                optim.step()
                optim.zero_grad()

                if is_master and not args.debug:
                    run.log({
                        **metrics.to_dict(),
                        "epoch": epoch + 1,
                        "step": step + 1,
                    })

                metrics.reset()

            progress_bar.set_description(
                f"Epoch {epoch + 1}, Loss: {output.loss.item():.4f}")


        logging.info(f"Epoch {epoch + 1} completed.")

        # Save checkpoint
        dist.barrier()

        if not args.debug and is_master:
            checkpoint_path = experiment_dir / f"model_epoch{epoch+1}.pt"
            logging.info(f"Saving model checkpoint to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Saved model checkpoint to {checkpoint_path}")


    logging.info(f"Training completed")
    logging.info(experiment_dir)


if __name__ == "__main__":
    main()