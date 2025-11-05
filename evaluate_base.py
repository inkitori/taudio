"""
Evaluate base TAudio model given a Hugging Face model id and dataset repo.
No experiment directory or checkpoints are used.
"""

import random
import torch
from dataset import infer_adapter_from_repository
import argparse
import logging
from tasks import create_task
from utils.metrics import AverageMetrics
import numpy as np
from transformers import set_seed
from models import create_adapter as create_model_adapter
from dataset import create_adapter as create_dataset_adapter
import wandb

def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Evaluate base TAudio using HF model id and dataset repo.")
    parser.add_argument('--model-id', type=str, required=True,
                        help='Hugging Face model id, e.g. Qwen/Qwen2.5-Omni-3B')
    parser.add_argument('--repository', type=str, required=True,
                        help='Hugging Face dataset repository to evaluate on')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate on')
    parser.add_argument('--seed', type=int, default=80,
                        help='Random seed')
    args = parser.parse_args()

    run = wandb.init(
        entity="taudio",
        project="Base Evaluations",
        name=f"[{args.model_id}][{args.repository}][{args.split}]",
        config={
            "model_id": args.model_id,
            "repository": args.repository,
            "split": args.split,
            "seed": args.seed,
        },
    )

    # Set random seed
    SEED = args.seed
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    set_seed(SEED)    

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    logging.info("Evaluating base model (no checkpoints).")

    # Build task
    task_kwargs = {
        'key': 'start',
    }
    task = create_task(task_type='SINGLE_WORD_TIMESTAMP', **task_kwargs)


    # Load dataset
    model_adapter = create_model_adapter(args.model_id, bidirectional_audio=False, dtype='auto', scaling_factor=1)
    model_adapter.base_model.to(torch.cuda.current_device())
    model_adapter.base_model.eval()

    dataset_adapter = create_dataset_adapter(
        infer_adapter_from_repository(args.repository),
        repository=args.repository,
        sampling_rate=model_adapter.sampling_rate,
        left_padding=0,
        key=task.key,
    )

    base_ds = dataset_adapter.load_split(args.split)
    # base_ds = base_ds.shuffle(seed=SEED)

    # Metrics aggregator (running averages)
    metrics = AverageMetrics()

    logging.info(f"Evaluating on {args.split} split")

    for example in base_ds:
        if task.skip_example(example, dataset_adapter):
            continue
        # Token-based evaluation
        token_metrics = task.evaluate_tokens_base(
            example=example,
            ds_adapter=dataset_adapter,
            model_adapter=model_adapter,
        )
        metrics.update_dict(token_metrics)

        run.log(metrics.to_dict())

    logging.info(metrics.to_dict())
    run.finish()

if __name__ == "__main__":
    main()
