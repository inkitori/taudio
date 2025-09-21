import argparse
import logging
import random

import torch
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
from transformers import set_seed
import numpy as np

from dataset.dataset import collate_fn, get_ds
from tasks import create_task
from taudio import TAudio
from utils.config_utils import (
    ConfigManager,
    flatten_config,
    relative_path_to_experiment_name,
    relative_path_to_project_name,
)
from utils.metrics import AverageMetrics

from accelerate import Accelerator, PartialState

def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Train TAudio model.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config file')
    parser.add_argument('--no-timestamp', action='store_true',
                        help='Don\'t add timestamp to output directory name')
    parser.add_argument('--debug', action='store_true',
                        help='Don\'t log to wandb or experiment directory, and don\'t save model checkpoints')
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

    world_size = PartialState().num_processes # 1 batch per device
    batch_size_per_device = training_config['effective_batch_size'] // world_size

    logging.info(f"World Size: {world_size}")
    logging.info(f"Batch size per device: {batch_size_per_device}")

    is_master = PartialState().is_main_process
    logging.info(f"Is master: {is_master}")

    torch.cuda.set_device(PartialState().device)
    logging.info(f"Using device: {torch.cuda.current_device()}")

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

    taudio_config = {
        **model_config,
        **loss_config,
        "task": task
    }

    # Create model
    model = TAudio(**taudio_config)
    model.train()

    accelerator = Accelerator()

    ds = get_ds(
        model_adapter=model.model_adapter,
        repository=dataset_config['repository'],
        split=dataset_config['split'],
        task=task,
        take_first=dataset_config.get('take_first', None),
        sharded=False,
    )

    accelerator.wait_for_everyone()

    dataloader = DataLoader(
        ds, 
        batch_size=batch_size_per_device, 
        drop_last=True,
        pin_memory=True,
        num_workers=8,
        collate_fn=collate_fn
    )

    optim = torch.optim.AdamW(model.parameters(), lr=training_config['learning_rate'])

    num_optim_steps = len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_optim_steps, eta_min=training_config['learning_rate'] * training_config['eta_min_scale'])

    model, optim, scheduler, dataloader = accelerator.prepare(model, optim, scheduler, dataloader)

    logging.info(f"Number of optimizer steps: {num_optim_steps}")
    logging.info(f"Dataloader length: {len(dataloader)}")

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
                "loss": loss.item(),
                "token_loss": token_loss.item(),
                "surrogate_loss": surrogate_loss.item(),
                "auxiliary_deviation": auxiliary_deviation.item(),
            })

            if is_master and not args.debug:
                run.log({
                    **metrics.to_dict(),
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "lr": scheduler.get_last_lr()[0],
                })

                metrics.reset()

                progress_bar.set_description(
                    f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


        logging.info(f"Epoch {epoch + 1} completed.")

        # Save checkpoint

        accelerator.wait_for_everyone()

        logging.info(f"Retrieving state dict")

        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = accelerator.get_state_dict(unwrapped_model)

        if is_master:
            checkpoint_path = experiment_dir / f"model_epoch{epoch+1}.pt"
            logging.info(f"Saving model checkpoint to {checkpoint_path}")
            torch.save(state_dict, checkpoint_path)



        # if not args.debug and is_master:

        # checkpoint_path = experiment_dir / f"model_epoch{epoch+1}.pt"
        # logging.info(f"Saving model checkpoint to {experiment_dir}")


            
        # logging.info(f"Saved model checkpoint to {checkpoint_path}")

        # logging.info(f"Unwrapping model")
        # unwrapped_model = accelerator.unwrap_model(model)
        # if is_master:
        #     logging.info(f"Saving model checkpoint to {experiment_dir}")
        #     torch.save(unwrapped_model.state_dict(), experiment_dir / f"model_epoch{epoch+1}.pt")

        # accelerator.save_model(model, experiment_dir, max_shard_size="80GB", safe_serialization=False)

        accelerator.wait_for_everyone()

    # messes with shell script if not protected
    if is_master:
        logging.info(f"Training completed. All outputs saved to: {experiment_dir}")


if __name__ == "__main__":
    main()