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
from utils.config_utils import ConfigManager
import logging
from tasks import create_task
from utils.metrics import AverageMetrics

def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Evaluate TAudio model.")
    parser.add_argument('--experiment', type=str, required=True,
                        help='Name of the experiment directory to evaluate')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Specific epoch to evaluate (defaults to latest)')
    parser.add_argument('--split', type=str, default='dev_clean',
                        help='Dataset split to evaluate on')
    parser.add_argument('--error-bound', type=float, default=0.1,
                        help='Error bound for considering predictions correct')
    parser.add_argument('--min-time', type=float, default=None,
                        help='Minimum time for considering predictions correct')
    parser.add_argument('--max-time', type=float, default=None,
                        help='Maximum time for considering predictions correct')
    args = parser.parse_args()

    # Initialize config manager
    config_manager = ConfigManager()

    # Find experiment directory
    experiment_dir = config_manager.get_experiment_path(args.experiment)
    if experiment_dir is None:
        # Try to find by base name
        experiment_dir = config_manager.find_latest_experiment(args.experiment)
        if experiment_dir is None:
            print(f"Experiment not found: {args.experiment}")
            print("Available experiments:")
            for exp in config_manager.list_experiments():
                print(f"  - {exp}")
            return

    print(f"Evaluating experiment: {experiment_dir}")

    # Load configuration
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)

    args.aux_output = config['loss']['surrogate_loss']
    args.token_output = config['loss']['token_loss']

    # Get model checkpoint
    checkpoint_path = config_manager.get_model_checkpoint(
        experiment_dir, args.epoch)
    if checkpoint_path is None:
        print(f"No checkpoint found in {experiment_dir}")
        return

    print(f"Using checkpoint: {checkpoint_path}")

    # Set random seed
    SEED = config['system']['seed']
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize wandb
    run = wandb.init(
        entity=config['wandb']['entity'],
        project="Eval",
        name=f"{experiment_dir.name}[{args.split}][epoch_{args.epoch}]",
        config={
            "experiment_name": config['experiment_name'],
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

    task = create_task(task_type=task_config['type'], **task_config.get('kwargs', {}))

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
        sampling_rate=model.adapter.sampling_rate,
    )
    base_ds = adapter.load_streaming_split(args.split)

    # Metrics aggregator (running averages)
    metrics = AverageMetrics()

    print(f"Evaluating on {args.split} split")

    model.eval()

    for example in base_ds:
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
