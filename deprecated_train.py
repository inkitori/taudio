import torch
import random
from tqdm.auto import tqdm
import bitsandbytes as bnb
import argparse
import logging
import wandb
from tasks import create_task
from taudio import TAudio
from utils.config_utils import ConfigManager, flatten_config, relative_path_to_experiment_name, relative_path_to_project_name
from utils.utils import get_dataset_length, patch_dataset_length
from dataset.dataset import get_ds, collate_fn
from utils.metrics import AverageMetrics
from dataset import infer_adapter_from_repository

def main():
    torch.autograd.set_detect_anomaly(True)
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

    # Create task
    task = create_task(task_type=task_config['type'], **task_config.get('kwargs', {}))

    taudio_config = {
        **model_config,
        **loss_config,
        "task": task
    }

    # Create model
    device = torch.device('cuda')

    model = TAudio(**taudio_config).to(device)
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
        batch_size=training_config['batch_size'],
        num_workers=system_config['dataloader_num_workers'],
        pin_memory=True,
        drop_last=True
    )

    # Setup optimizer and scheduler
    steps_per_epoch = dataset_length // training_config['batch_size']
    total_optimizer_steps = (
        steps_per_epoch * training_config['epochs']) // training_config['grad_accumulation_steps']

    if training_config['optim_8bit']:
        logging.info("Using AdamW8bit optimizer")
        optim = bnb.optim.AdamW8bit(
            model.parameters(), lr=training_config['learning_rate'])
    else:
        logging.info("Using AdamW optimizer")
        optim = bnb.optim.AdamW(
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

        run = wandb.init(
            entity=config['wandb']['entity'],
            project=project_name, 
            name=experiment_name,
            config=flattened_config,
        )
        
    logging.info(f"Using device: {torch.device('cuda')}")

    # Training loop
    for epoch in range(training_config['epochs']):
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch + 1}",
        )

        metrics = AverageMetrics()

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(**batch)

            logging.info(f"Done with forward pass")

            loss = output.loss
            token_loss = output.token_loss
            surrogate_loss = output.surrogate_loss
            auxiliary_deviation = output.auxiliary_deviation

            metrics.update_dict({
                "loss": loss.item(),
                "token_loss": token_loss.item(),
                "surrogate_loss": surrogate_loss.item(),
                "auxiliary_deviation": auxiliary_deviation,
            })
            logging.info(f"Done with metrics")

            scaled_loss = loss / training_config['grad_accumulation_steps']
            logging.info(f"Scaled loss: {scaled_loss}")
            scaled_loss.backward()
            logging.info(f"Done with backward pass")

            if (step + 1) % training_config['grad_accumulation_steps'] == 0:
                optim.step()
                logging.info(f"Done with optimizer step")
                optim.zero_grad()
                logging.info(f"Done with optimizer zero grad")
                scheduler.step()
                logging.info(f"Done with scheduler step")

                # Get memory stats before clearing cache
                if torch.cuda.is_available():
                    allocated_memory = torch.cuda.memory_allocated() / 1e9  # GB
                    reserved_memory = torch.cuda.memory_reserved() / 1e9  # GB
                    max_memory = torch.cuda.max_memory_allocated() / 1e9  # GB

                logging.info(
                    f"Step {step + 1}, Average Loss: {loss.item():.4f}, "
                    f"GPU Mem: {allocated_memory:.2f}/{reserved_memory:.2f}/{max_memory:.2f} GB (allocated/reserved/max)")

                if not args.debug:
                    run.log({
                        **metrics.to_dict(),
                        "step": step + 1,
                        "epoch": epoch + 1,
                        "learning_rate": scheduler.get_last_lr()[0],
                    })

                metrics.reset()

            progress_bar.set_description(
                f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        logging.info(f"Epoch {epoch + 1} completed.")

        # Save checkpoint
        if ((not args.debug and system_config.get('save_checkpoints', True)) or epoch == training_config['epochs'] - 1):
            checkpoint_path = experiment_dir / f"model_epoch{epoch + 1}.pt"
            logging.info(f"Saving model to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)

            logging.info(f"Model saved to {checkpoint_path}")


    # Log final experiment directory to wandb
    if not args.debug:
        run.log({"experiment_directory": str(experiment_dir)})
        run.finish()

    logging.info(f"Training completed. All outputs saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
