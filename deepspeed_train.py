import argparse
import logging
import random

import deepspeed
import torch
from tqdm.auto import tqdm

from dataset.dataset import get_ds, collate_fn
from tasks import create_task
from taudio import TAudio
from utils.config_utils import (
    ConfigManager,
    relative_path_to_experiment_name,
    relative_path_to_project_name,
)
from utils.metrics import AverageMetrics

def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Train TAudio model.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config file')
    parser.add_argument('--debug', action='store_true',
                        help='Don\'t log to wandb or experiment directory, and don\'t save model checkpoints')
    parser.add_argument('--local_rank', type=int, default=0)

    parser = deepspeed.add_config_arguments(parser)

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
        experiment_dir = config_manager.create_experiment_dir(args.config)
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
    model = TAudio(**taudio_config)
    model.train()

    ds = get_ds(
        model_adapter=model.adapter,
        repository=dataset_config['repository'],
        split=dataset_config['split'],
        task=task,
        take_first=dataset_config.get('take_first', None),
    )

    model_engine, _, dataloader, _ = deepspeed.initialize(
        args=args, 
        model=model, 
        model_parameters=model.parameters(),
        training_data=ds,
        collate_fn=collate_fn
    )

    # Training loop
    for epoch in range(training_config['epochs']):
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch + 1}",
        )

        metrics = AverageMetrics()

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            output = model_engine(**batch)

            local_loss = output.loss
            local_token_loss = output.token_loss
            local_surrogate_loss = output.surrogate_loss
            local_auxiliary_deviation = output.auxiliary_deviation

            model_engine.backward(local_loss)

            model_engine.step()
            progress_bar.set_description(
                f"Epoch {epoch + 1}, Loss: {local_loss.item():.4f}")

        logging.info(f"Epoch {epoch + 1} completed.")

        # Save checkpoint
        if not args.debug:
            logging.info(f"Saving model to {experiment_dir}")
            model_engine.save_checkpoint(experiment_dir)
            logging.info(f"Model saved to {experiment_dir}")

    logging.info(f"Training completed. All outputs saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
