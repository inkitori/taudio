import torch
import random
from tqdm.auto import tqdm
import bitsandbytes as bnb
import argparse
import logging
from accelerate import Accelerator, PartialState
from accelerate.utils import DataLoaderConfiguration, GradientAccumulationPlugin

from tasks import create_task
from taudio import TAudio
from utils.config_utils import ConfigManager, flatten_config, relative_path_to_experiment_name, relative_path_to_project_name
from utils.utils import get_dataset_length, patch_dataset_length
from dataset.dataset import get_ds, collate_fn
from utils.metrics import AverageMetrics
from dataset import infer_adapter_from_repository

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

    # Set random seed
    random.seed(system_config['seed'])
    torch.manual_seed(system_config['seed'])

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

    logging.info("Number of processes: ", PartialState().num_processes)
    batch_size = PartialState().num_processes
    gradient_accumulation_steps = training_config['effective_batch_size'] // batch_size

    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    accelerator_dataloader_config = DataLoaderConfiguration(
        dispatch_batches=True,
        split_batches=True,
    )

    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=gradient_accumulation_steps,
        adjust_scheduler=False,
    )

    accelerator = Accelerator(
        log_with="wandb", 
        dataloader_config=accelerator_dataloader_config, 
        gradient_accumulation_plugin=gradient_accumulation_plugin
    )

    logging.info(f"Using accelerator: {accelerator}")
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


    ds = get_ds(
        model_adapter=model.adapter,
        repository=dataset_config['repository'],
        split=dataset_config['split'],
        task=task,
        take_first=dataset_config.get('take_first', None),
    )

    # Create dataloader


    dataset_length = dataset_config.get('take_first', get_dataset_length(
        dataset_config['repository'], dataset_config['split']))
    patch_dataset_length(ds, dataset_length)

    dataloader = torch.utils.data.DataLoader(
        ds,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=system_config['dataloader_num_workers'],
        # pin_memory=True,
        drop_last=True
    )

    # Setup optimizer and scheduler
    steps_per_epoch = dataset_length // batch_size
    total_optimizer_steps = (
        steps_per_epoch * training_config['epochs']) // gradient_accumulation_steps

    if training_config['optim_8bit']:
        logging.info("Using AdamW8bit optimizer")
        optim = bnb.optim.AdamW8bit(
            model.parameters(), lr=training_config['learning_rate'])
    else:
        logging.info("Using AdamW optimizer")
        optim = torch.optim.AdamW(
            model.parameters(), lr=training_config['learning_rate'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=total_optimizer_steps,
        eta_min=training_config['learning_rate'] *
        training_config['eta_min_scale']
    )

    if not args.debug:
        # Initialize wandb
        flattened_config = flatten_config(config)

        accelerator.init_trackers(
            project_name=project_name, 
            config=flattened_config,
            init_kwargs={"wandb": {"name": experiment_name, "entity": config['wandb']['entity']}}
        )
        
    logging.info(f"Using device: {accelerator.device}")

    model, optim, dataloader, scheduler = accelerator.prepare(model, optim, dataloader, scheduler)

    # Training loop
    for epoch in range(training_config['epochs']):
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch + 1}",
            disable=(not accelerator.is_main_process)
        )

        metrics = AverageMetrics()

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                output = model(**batch)

                local_loss = output.loss
                local_token_loss = output.token_loss
                local_surrogate_loss = output.surrogate_loss
                local_auxiliary_deviation = torch.tensor(output.auxiliary_deviation, dtype=local_loss.dtype)

                loss = accelerator.gather(local_loss).mean().item()
                token_loss = accelerator.gather(local_token_loss).mean().item()
                surrogate_loss = accelerator.reduce(local_surrogate_loss).mean().item()
                auxiliary_deviation = accelerator.reduce(local_auxiliary_deviation).mean().item()

                metrics.update_dict({
                    "loss": loss,
                    "token_loss": token_loss,
                    "surrogate_loss": surrogate_loss,
                    "auxiliary_deviation": auxiliary_deviation,
                })

                accelerator.backward(loss)

                optim.step()
                optim.zero_grad()
                scheduler.step()

                if not args.debug:
                    accelerator.log({
                        **metrics.to_dict(),
                        "step": step + 1,
                        "epoch": epoch + 1,
                        "learning_rate": scheduler.get_last_lr()[0],
                    })

                metrics.reset()

            progress_bar.set_description(
                f"Epoch {epoch + 1}, Loss: {loss:.4f}")

        logging.info(f"Epoch {epoch + 1} completed.")

        # Save checkpoint
        if ((not args.debug and system_config.get('save_checkpoints', True)) or epoch == training_config['epochs'] - 1) and accelerator.is_main_process:
            checkpoint_path = experiment_dir / f"model_epoch{epoch + 1}.pt"

            # logging.info(f"Saving model to {checkpoint_path}")
            # accelerator.save_model(model, checkpoint_path)

            # logging.info(f"Model saved to {checkpoint_path}")

            logging.info(f"Unwrapping model")
            unwrapped_model = accelerator.unwrap_model(model)

            logging.info(f"Saving model to {checkpoint_path}")
            accelerator.save(unwrapped_model.state_dict(), checkpoint_path)

            logging.info(f"Model saved to {checkpoint_path}")

        accelerator.wait_for_everyone()

    # Log final experiment directory to wandb
    if not args.debug:
        accelerator.log({"experiment_directory": str(experiment_dir)})
        accelerator.end_training()

    logging.info(f"Training completed. All outputs saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
