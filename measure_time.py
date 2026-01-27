"""
Measures average inference time per batch for TAudio models.
Supports both token loss (text generation) and poisson/surrogate loss (auxiliary outputs).
Works with timestamp_single_any and timestamp_all tasks.
"""

import argparse
import logging
import time
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import collate_fn, get_benchmark_ds
from tasks import create_task
from taudio import TAudio
from utils.config_utils import ConfigManager
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
) -> List[float]:
    times = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Token Inference")):
        if num_batches is not None and batch_idx >= num_batches:
            break
        
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
    
    return times


def measure_poisson_inference_time(
    model: TAudio,
    dataloader: DataLoader,
    num_batches: int,
) -> List[float]:
    """
    Measure inference time for poisson/surrogate loss models.
    Inference involves a forward pass with inference=True to get auxiliary predictions.
    """
    times = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Poisson Inference")):
        if num_batches is not None and batch_idx >= num_batches:
            break
        
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
    
    return times


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
        help="Number of batches to measure (default: 50)"
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
        help="Number of warmup batches before timing (default: 5)"
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
    
    dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=system_config.get('dataloader_num_workers', 1),
    )
    
    logging.info(f"Dataset loaded with {len(ds)} examples")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Number of batches to measure: {args.num_batches}")
    logging.info(f"Warmup batches: {args.warmup_batches}")
    
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
    
    # Reset dataloader for actual measurements
    dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=system_config.get('dataloader_num_workers', 1),
    )
    
    if token_loss:
        times = measure_token_inference_time(model, dataloader, args.num_batches, task_type)
        inference_type = "Token Generation"
    else:
        times = measure_poisson_inference_time(model, dataloader, args.num_batches)
        inference_type = "Poisson/Surrogate"
    
    # Calculate statistics
    if len(times) > 0:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Calculate std dev
        variance = sum((t - avg_time) ** 2 for t in times) / len(times)
        std_time = variance ** 0.5
        
        # Calculate throughput
        avg_throughput = args.batch_size / avg_time  # samples per second
        
        print("\n" + "=" * 60)
        print(f"INFERENCE TIMING RESULTS")
        print("=" * 60)
        print(f"Config: {args.config}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Task Type: {task_type}")
        print(f"Inference Type: {inference_type}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Number of Batches Measured: {len(times)}")
        print("-" * 60)
        print(f"Average Time per Batch: {avg_time * 1000:.2f} ms")
        print(f"Std Dev: {std_time * 1000:.2f} ms")
        print(f"Min Time: {min_time * 1000:.2f} ms")
        print(f"Max Time: {max_time * 1000:.2f} ms")
        print("-" * 60)
        print(f"Average Throughput: {avg_throughput:.2f} samples/second")
        print(f"Average Time per Sample: {(avg_time / args.batch_size) * 1000:.2f} ms")
        print("=" * 60)
    else:
        logging.error("No timing measurements collected!")


if __name__ == "__main__":
    main()

