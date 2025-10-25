"""
Evaluation script for TAudio model with clean configuration management.
"""

import random
import torch
from dataset import create_adapter, infer_adapter_from_repository
from taudio import TAudio
import wandb
import argparse
from pathlib import Path
from utils.config_utils import ConfigManager, relative_path_to_experiment_name, relative_path_to_project_name
import logging
from tasks import create_task
from utils.metrics import AverageMetrics
import numpy as np
from transformers import set_seed
def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Evaluate TAudio model.")
    parser.add_argument('--experiment', type=str, required=True,
                        help='Name of the experiment directory to evaluate')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Specific epoch to evaluate (defaults to latest)')
    parser.add_argument('--split', type=str, default='test_clean',
                        help='Dataset split to evaluate on')
    parser.add_argument('--error-bound', type=float, default=0.02,
                        help='Error bound for considering predictions correct')
    parser.add_argument('--min-time', type=float, default=None,
                        help='Minimum time for considering predictions correct')
    parser.add_argument('--max-time', type=float, default=None,
                        help='Maximum time for considering predictions correct')
    args = parser.parse_args()

    # Initialize config manager
    config_manager = ConfigManager()

    # Find experiment directory
    experiment_dir = args.experiment
    experiment_name = relative_path_to_experiment_name(experiment_dir, eval=True)
    project_name = relative_path_to_project_name(experiment_dir, eval=True)

    logging.info(f"Evaluating experiment: {experiment_dir}")

    # Load configuration
    experiment_dir = Path(experiment_dir)
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)

    args.aux_output = config['loss']['surrogate_loss']
    args.token_output = config['loss']['token_loss']

    # Get model checkpoint
    checkpoint_path = config_manager.get_model_checkpoint(
        experiment_dir, args.epoch)
    logging.info(f"Checkpoint path: {checkpoint_path}")
    if checkpoint_path is None:
        raise ValueError(f"No checkpoint found in {experiment_dir}")
        return

    logging.info(f"Using checkpoint: {checkpoint_path}")

    # Set random seed
    SEED = config['system']['seed']
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    set_seed(SEED)    

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Initialize wandb
    run = wandb.init(
        entity=config['wandb']['entity'],
        project=project_name,
        name=f"{experiment_name}[{args.split}][epoch_{args.epoch if args.epoch is not None else 'latest'}][{args.min_time}-{args.max_time}][bound_{args.error_bound}]",
        config={
            "experiment_name": experiment_name,
            "checkpoint_path": str(checkpoint_path),
            "split": args.split,
            "aux_output": args.aux_output,
            "token_output": args.token_output,
            "error_bound": args.error_bound,
            "min_time": args.min_time,
            "max_time": args.max_time,
            **config
        }
    )

    # Load model
    model_config = config['model']
    loss_config = config['loss']
    dataset_config = config['dataset']
    task_config = config['task']

    task_kwargs = task_config.get('kwargs', {})
    task_kwargs['min_time'] = args.min_time
    task_kwargs['max_time'] = args.max_time

    task = create_task(task_type=task_config['type'], **task_kwargs)

    taudio_config = {
        **model_config,
        **loss_config,
        "task": task
    }

    model = TAudio(**taudio_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Load dataset
    adapter = create_adapter(
        infer_adapter_from_repository(dataset_config['repository']),
        repository=dataset_config['repository'],
        sampling_rate=model.model_adapter.sampling_rate,
        left_padding=dataset_config.get('left_padding', 0),
        key=task.key,
    )
    task.rounding_factor = adapter.timestamp_rounding_factor()

    base_ds = adapter.load_split(args.split)
    # base_ds = base_ds.shuffle(seed=SEED)

    # Metrics aggregator (running averages)
    metrics = AverageMetrics()

    logging.info(f"Evaluating on {args.split} split")

    model.eval()

    for example in base_ds:
        if task.skip_example(example, adapter):
            continue
        # Token-based evaluation
        if args.token_output:
            token_metrics = task.evaluate_tokens(
                example=example,
                ds_adapter=adapter,
                model=model,
                error_bound=args.error_bound,
            )
            if token_metrics is None:
                continue
            metrics.update_dict(token_metrics)

        # Auxiliary-head evaluation
        if args.aux_output:
            aux_metrics = task.evaluate_auxiliary_outputs(
                example=example,
                ds_adapter=adapter,
                model=model,
                error_bound=args.error_bound,
            )
            if aux_metrics is None:
                continue
            # Accuracy over all aux examples
            metrics.update_dict(aux_metrics)

        run.log(metrics.to_dict())

    run.finish()


if __name__ == "__main__":
    main()
