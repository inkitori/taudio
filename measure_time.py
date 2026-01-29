"""
Measures average inference time per batch for TAudio models.
Supports both token loss (text generation) and poisson/surrogate loss (auxiliary outputs).
Works with timestamp_single_any and timestamp_all tasks.
"""

import argparse
import logging
import time
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb

from dataset.dataset import collate_fn, get_benchmark_ds
from tasks import create_task
from taudio import TAudio
from utils.config_utils import (
    ConfigManager,
    flatten_config,
    relative_path_to_experiment_name,
    relative_path_to_project_name,
)
from utils.poisson import infer_timestamps_benchmark


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def measure_token_inference_time(
    model: TAudio,
    dataloader: DataLoader,
    num_batches: int,
    task_type: str,
    bucket_label: str,
) -> Tuple[List[float], List[int]]:
    times = []
    batch_sizes = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc=f"Token Inference [{bucket_label}]")):
        if num_batches is not None and batch_idx >= num_batches:
            break
        
        batch_sizes.append(batch["input_ids"].size(0))

        # Move batch to device
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}

        # logging.info(batch)
        
        # Prepare inputs for generation (exclude labels and audio_labels)
        gen_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "input_features": batch["input_features"],
            "feature_attention_mask": batch["feature_attention_mask"],
        }
        
        # Synchronize before timing
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            # logging.info("DECODED OUTPUT")
            output = model.model_adapter.generate_batch(**gen_inputs, max_new_tokens=4096, decode_tokens=True)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        logging.info(output)
        
        times.append(end_time - start_time)
    
    return times, batch_sizes


def measure_poisson_inference_time(
    model: TAudio,
    dataloader: DataLoader,
    num_batches: int,
    bucket_label: str,
) -> Tuple[List[float], List[int]]:
    """
    Measure inference time for poisson/surrogate loss models.
    Inference involves a forward pass with inference=True to get auxiliary predictions.
    """
    times = []
    batch_sizes = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc=f"Poisson Inference [{bucket_label}]")):
        if num_batches is not None and batch_idx >= num_batches:
            break
        
        batch_sizes.append(batch["input_ids"].size(0))

        # Move batch to device
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        
        # Synchronize before timing
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        batch_size = batch['input_ids'].size(0)

        with torch.no_grad():
            outputs = model.model_adapter(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                input_features=batch["input_features"],
                feature_attention_mask=batch["feature_attention_mask"],
                output_hidden_states=True
            )

            hidden_states = outputs.hidden_states[model.audio_layer]

            for example in range(batch_size):
                audio_hidden_states = hidden_states[example][batch['input_ids'][example] == model.model_adapter.audio_id] # (num_audio_tokens, hidden_dim)
                example_audio_logits = model.linear(audio_hidden_states).flatten() # (num_audio_tokens * scaling_factor,)
                
                example_audio_labels = batch['audio_labels'][example]

                if (example_audio_labels == -100).any():
                    neg_100_idx = (example_audio_labels == -100).nonzero(as_tuple=True)[0][0].item()
                    example_audio_logits = example_audio_logits[:neg_100_idx]
                    example_audio_labels = example_audio_labels[:neg_100_idx]

                num_pred = (batch['audio_labels'][example] == 1).sum().item()

                predictions = infer_timestamps_benchmark(num_pred, example_audio_logits.cpu().float().detach().numpy(), model.model_adapter.embedding_to_frame_adjusted_milliseconds, 20)
            
                # logging.info("GROUND TRUTH")
                # logging.info(torch.where(batch['audio_labels'][example] == 1)[0] / (model.model_adapter.seconds_to_embedding * model.model_adapter.scaling_factor))
                # logging.info("PREDICTED")
                # logging.info(predictions / (model.model_adapter.seconds_to_embedding * model.model_adapter.scaling_factor))

        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
    
    return times, batch_sizes


EVENT_BUCKETS = [
    {"label": "1-5 events", "min": 1, "max": 5, "key": "1-5_events"},
    {"label": "6-10 events", "min": 6, "max": 10, "key": "6-10_events"},
    {"label": "11-15 events", "min": 11, "max": 15, "key": "11-15_events"},
    {"label": "16-20 events", "min": 16, "max": 20, "key": "16-20_events"},
    {"label": "21-25 events", "min": 21, "max": 25, "key": "21-25_events"},
    {"label": "26+ events", "min": 26, "max": None, "key": "26plus_events"},
]


def _normalize_num_events(value: torch.Tensor) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _bucket_for_num_events(num_events: int) -> Dict[str, str]:
    for bucket in EVENT_BUCKETS:
        if bucket["max"] is None:
            if num_events >= bucket["min"]:
                return bucket
        elif bucket["min"] <= num_events <= bucket["max"]:
            return bucket
    raise ValueError(f"Unexpected num_events: {num_events}")


def _infer_num_events_from_example(example: Dict[str, torch.Tensor]) -> int:
    if "num_events" in example:
        return _normalize_num_events(example["num_events"])
    if "audio_labels" in example:
        audio_labels = example["audio_labels"]
        if torch.is_tensor(audio_labels):
            return int((audio_labels == 1).sum().item())
        return int(sum(1 for v in audio_labels if v == 1))
    raise KeyError("Example has neither 'num_events' nor 'audio_labels'.")


def _build_bucket_indices(ds) -> Dict[str, List[int]]:
    indices = {bucket["label"]: [] for bucket in EVENT_BUCKETS}
    for idx in range(len(ds)):
        example = ds[idx]
        num_events = _infer_num_events_from_example(example)
        bucket = _bucket_for_num_events(num_events)
        indices[bucket["label"]].append(idx)
    return indices


def _create_bucket_dataloaders(
    ds,
    bucket_indices: Dict[str, List[int]],
    *,
    batch_size: int,
    collate_fn,
    num_workers: int,
) -> Dict[str, DataLoader]:
    dataloaders = {}
    for bucket in EVENT_BUCKETS:
        label = bucket["label"]
        indices = bucket_indices[label]
        if not indices:
            continue
        dataloaders[label] = DataLoader(
            Subset(ds, indices),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
    return dataloaders


def main():
    parser = argparse.ArgumentParser(
        description="Measure TAudio inference time per batch"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pt checkpoint file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config YAML file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="Number of batches to measure"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    model_config = config['model']
    loss_config = config['loss']
    dataset_config = config['dataset']
    task_config = config['task']
    system_config = config.get('system', {})
    
    # Determine loss type
    token_loss = loss_config.get('token_loss', False)
    poisson_loss = loss_config.get('poisson_loss', False)
    surrogate_loss = loss_config.get('surrogate_loss', False)
    
    logging.info(f"Loss configuration: token_loss={token_loss}, poisson_loss={poisson_loss}, surrogate_loss={surrogate_loss}")
    
    if not token_loss and not (poisson_loss or surrogate_loss):
        raise ValueError("Config must have either token_loss or poisson_loss/surrogate_loss enabled")
    
    # Create task
    task = create_task(task_type=task_config['type'], **task_config.get('kwargs', {}))
    task_type = task_config['type']
    
    logging.info(f"Task type: {task_type}")
    
    # Build model configuration
    taudio_config = {
        **model_config,
        **loss_config,
        "task": task
    }
    
    # Create model
    logging.info("Creating TAudio model...")
    model = TAudio(**taudio_config)
    
    # Load checkpoint
    logging.info(f"Loading checkpoint from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    logging.info("Checkpoint loaded successfully")
    
    # Move model to GPU and set to eval mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    logging.info(f"Model moved to {device}")

    # Initialize wandb
    run = None
    wandb_config = config.get("wandb", {})
    if wandb_config:
        experiment_name = relative_path_to_experiment_name(args.config, eval=False)
        project_name = relative_path_to_project_name(args.config, eval=False)
        run = wandb.init(
            entity=wandb_config.get("entity"),
            project="measure_time",
            name=f"[{model_config['model_id']}][{task_config}][{args.split}][bs_{args.batch_size}][num_batches_{args.num_batches}]",
            config={
                **flatten_config(config),
                "measure_time": {
                    "checkpoint": args.checkpoint,
                    "config": args.config,
                    "batch_size": args.batch_size,
                    "num_batches": args.num_batches,
                    "split": args.split,
                },
            },
        )
    
    # Create dataset and dataloader
    logging.info("Loading dataset...")
    ds, ds_adapter = get_benchmark_ds(
        model_adapter=model.model_adapter,
        repository=dataset_config['repository'],
        split=args.split,
        task=task,
        eval_mode=token_loss
    )
    
    # Pre-select indices for timestamp_single_any task if needed
    if hasattr(task, 'select_indices'):
        ds = task.select_indices(ds, ds_adapter, args.split)
    
    logging.info(f"Dataset loaded with {len(ds)} examples")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Number of batches to measure: {args.num_batches}")
    
    # Warmup runs
    # logging.info("Running warmup batches...")
    # warmup_iter = iter(dataloader)
    # for _ in range(args.warmup_batches):
    #     try:
    #         batch = next(warmup_iter)
    #         batch = {k: v.to(device) for k, v in batch.items()}
            
    #         with torch.no_grad():
    #             if token_loss:
    #                 gen_inputs = {
    #                     "input_ids": batch["input_ids"],
    #                     "attention_mask": batch["attention_mask"],
    #                     "input_features": batch["input_features"],
    #                     "feature_attention_mask": batch["feature_attention_mask"],
    #                 }
    #                 if task_type == "ALL_TIMESTAMPS":
    #                     logging.info(model.model_adapter.generate_batch(**gen_inputs, max_new_tokens=4096))
    #                 else:
    #                     logging.info(model.generate(**{k: v[0:1] for k, v in gen_inputs.items()}))
    #             else:
    #                 _ = model(
    #                     input_ids=batch["input_ids"],
    #                     attention_mask=batch["attention_mask"],
    #                     input_features=batch["input_features"],
    #                     feature_attention_mask=batch["feature_attention_mask"],
    #                     inference=True,
    #                     true_inference=True,
    #                 )
    #     except StopIteration:
    #         logging.warning("Not enough data for warmup, resetting iterator")
    #         warmup_iter = iter(dataloader)
    
    torch.cuda.synchronize()
    logging.info("Warmup complete")
    
    # Measure inference time
    logging.info("Starting inference timing measurements...")
    
    # Build bucketed dataloaders based on number of events
    logging.info("Building event-count buckets...")
    bucket_indices = _build_bucket_indices(ds)
    dataloaders = _create_bucket_dataloaders(
        ds,
        bucket_indices,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=system_config.get('dataloader_num_workers', 1),
    )

    inference_type = "Token Generation" if token_loss else "Poisson/Surrogate"

    any_measurements = False
    for bucket in EVENT_BUCKETS:
        bucket_label = bucket["label"]
        bucket_key = bucket["key"]
        if bucket_label not in dataloaders:
            logging.info(f"Skipping bucket '{bucket_label}' (no samples)")
            continue

        dataloader = dataloaders[bucket_label]
        if token_loss:
            times, batch_sizes = measure_token_inference_time(
                model, dataloader, args.num_batches, task_type, bucket_label
            )
        else:
            times, batch_sizes = measure_poisson_inference_time(
                model, dataloader, args.num_batches, bucket_label
            )

        if len(times) == 0:
            logging.warning(f"No timing measurements collected for bucket '{bucket_label}'")
            continue

        any_measurements = True
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        variance = sum((t - avg_time) ** 2 for t in times) / len(times)
        std_time = variance ** 0.5

        total_samples = sum(batch_sizes)
        total_time = sum(times)
        avg_throughput = total_samples / total_time if total_time > 0 else 0.0
        avg_time_per_sample = (total_time / total_samples) * 1000 if total_samples > 0 else 0.0

        if run is not None:
            run.log({
                f"timing/{bucket_key}/avg_ms": avg_time * 1000,
                f"timing/{bucket_key}/std_ms": std_time * 1000,
                f"timing/{bucket_key}/min_ms": min_time * 1000,
                f"timing/{bucket_key}/max_ms": max_time * 1000,
                f"timing/{bucket_key}/avg_s": avg_time,
                f"timing/{bucket_key}/throughput_samples_per_s": avg_throughput,
                f"timing/{bucket_key}/avg_ms_per_sample": avg_time_per_sample,
                f"timing/{bucket_key}/batches_measured": len(times),
                f"timing/{bucket_key}/samples_measured": total_samples,
                f"timing/{bucket_key}/inference_type": inference_type,
                f"timing/{bucket_key}/times_hist": wandb.Histogram(times),
            })

        print("\n" + "=" * 60)
        print("INFERENCE TIMING RESULTS")
        print("=" * 60)
        print(f"Config: {args.config}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Task Type: {task_type}")
        print(f"Inference Type: {inference_type}")
        print(f"Bucket: {bucket_label}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Number of Batches Measured: {len(times)}")
        print(f"Number of Samples Measured: {total_samples}")
        print("-" * 60)
        print(f"Average Time per Batch: {avg_time * 1000:.2f} ms")
        print(f"Std Dev: {std_time * 1000:.2f} ms")
        print(f"Min Time: {min_time * 1000:.2f} ms")
        print(f"Max Time: {max_time * 1000:.2f} ms")
        print("-" * 60)
        print(f"Average Throughput: {avg_throughput:.2f} samples/second")
        print(f"Average Time per Sample: {avg_time_per_sample:.2f} ms")
        print("=" * 60)

    if not any_measurements:
        logging.error("No timing measurements collected across buckets!")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()

