import argparse
import logging
import time
from typing import Dict, List, Tuple

import torch

from tasks import create_task
from utils.config_utils import ConfigManager
from models import create_adapter
from dataset import create_adapter as create_ds_adapter, infer_adapter_from_repository
from taudio import TAudio
from pathlib import Path


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def prefill_with_kv(adapter, inputs: Dict[str, torch.Tensor]):
    """
    Runs a prefill forward pass to build KV cache for the full prompt (text + audio),
    returning the model outputs (including past_key_values) and the input length.
    """
    device = next(adapter.parameters()).device
    # Ensure tensors are on the correct device
    model_inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "input_features": inputs["input_features"].to(device),
        "feature_attention_mask": inputs["feature_attention_mask"].to(device),
        "use_cache": True,
    }

    # Maintain bidirectional audio mask if enabled for the adapter
    with adapter.bidirectional_audio_context(model_inputs["input_ids"]):
        outputs = adapter(**model_inputs)
    return outputs, model_inputs["input_ids"].shape[1], model_inputs


@torch.inference_mode()
def measure_generation_time_kv(
    adapter,
    processor,
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int = 32,
) -> Tuple[float, int, str]:
    """
    Measures only the wall-clock time of generating new tokens using an explicit KV-cache loop.
    Excludes the prefill time.
    Returns (elapsed_seconds, num_generated_tokens, decoded_text).
    """
    device = next(adapter.parameters()).device

    # 1) Prefill (NOT TIMED)
    prefill_outputs, input_len, prefill_inputs = prefill_with_kv(adapter, inputs)
    past_key_values = prefill_outputs.past_key_values

    # The first step uses the last prompt token
    next_token = prefill_inputs["input_ids"][:, -1:]
    eos_token_id = processor.tokenizer.eos_token_id

    generated = []
    elapsed = 0.0

    # 2) Decode loop (TIMED)
    _cuda_sync()
    t0 = time.perf_counter()
    with adapter.bidirectional_audio_context(prefill_inputs["input_ids"]):
        for _ in range(max_new_tokens):
            step_inputs = {
                "input_ids": next_token,  # (1, 1)
                "past_key_values": past_key_values,
            }
            # For decoding steps, we do not resend audio features; KV holds the context.
            outputs = adapter.base_model(**step_inputs)
            logits = outputs.logits[:, -1, :]  # (1, vocab)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (1, 1)
            generated.append(next_token.item())
            past_key_values = outputs.past_key_values

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    _cuda_sync()
    elapsed = time.perf_counter() - t0

    decoded_text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return elapsed, len(generated), decoded_text


@torch.inference_mode()
def measure_generation_time_no_kv(
    adapter,
    processor,
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int = 32,
) -> Tuple[float, int, str]:
    """
    Measures wall-clock time of generation using the model's built-in generate (includes context/prefill).
    Returns (elapsed_seconds, num_generated_tokens, decoded_text).
    """
    device = next(adapter.parameters()).device
    gen_kwargs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "input_features": inputs["input_features"].to(device),
        "feature_attention_mask": inputs["feature_attention_mask"].to(device),
        "max_new_tokens": max_new_tokens,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "do_sample": False,
        "use_cache": False,
    }
    prompt_len = inputs["input_ids"].shape[1]

    _cuda_sync()
    t0 = time.perf_counter()
    with adapter.bidirectional_audio_context(gen_kwargs["input_ids"]):
        tokens = adapter.base_model.generate(**gen_kwargs)
    _cuda_sync()
    elapsed = time.perf_counter() - t0

    out_ids = tokens[0]
    # drop EOS if present at end
    gen_ids = out_ids[prompt_len:-1] if out_ids.shape[0] > prompt_len else out_ids[prompt_len:]
    decoded_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    return elapsed, int(gen_ids.shape[0]), decoded_text


@torch.inference_mode()
def measure_surrogate_time(
    taudio_model,
    task,
    adapter,
    inputs: Dict[str, torch.Tensor],
    poisson_loss: bool,
    class_weighting: bool,
) -> float:
    """
    Measures only:
      - passing last-layer audio embeddings through the linear layer
      - then running Poisson or Bernoulli timestamp inference via task.calculate_loss
    Excludes the forward pass to compute hidden states.
    """
    device = next(adapter.parameters()).device
    # 1) Run a forward to obtain hidden states (NOT TIMED)
    with adapter.bidirectional_audio_context(inputs["input_ids"]):
        outputs = adapter(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            input_features=inputs["input_features"].to(device),
            feature_attention_mask=inputs["feature_attention_mask"].to(device),
            output_hidden_states=True,
        )

    # 2) Slice out audio token hidden states and compute logits with linear (TIMED)
    hidden_states = outputs.hidden_states[taudio_model.audio_layer]  # (1, seq_len, hidden_dim)
    input_ids_b = inputs["input_ids"].to(device)
    audio_token_mask = (input_ids_b == adapter.audio_id)[0]  # (seq_len,)
    audio_hidden_states = hidden_states[0][audio_token_mask]  # (num_audio_tokens, hidden_dim)

    _cuda_sync()
    t0 = time.perf_counter()
    audio_logits = taudio_model.linear(audio_hidden_states).flatten().unsqueeze(0)  # (1, num_audio_tokens*scaling)

    # Build minimal inference labels/mask for calculate_loss
    audio_labels = torch.zeros_like(audio_logits)
    audio_labels_frame_mask = torch.where(audio_labels == -100, 0, 1)

    # Run the task's inference (Poisson/Bernoulli) on logits
    task.calculate_loss(
        audio_logits=audio_logits,
        audio_labels=audio_labels,
        audio_labels_frame_mask=audio_labels_frame_mask,
        model_adapter=adapter,
        use_poisson_loss=poisson_loss,
        class_weighting=class_weighting,
    )
    _cuda_sync()
    elapsed = time.perf_counter() - t0
    return elapsed


def main():
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="Measure inference timing for token and surrogate paths.")
    parser.add_argument("--config", type=str, required=False, help="Path to config YAML.")
    parser.add_argument("--experiment", type=str, required=False, help="Path to experiment directory (loads config.yaml and latest or specified checkpoint).")
    parser.add_argument("--epoch", type=int, default=None, help="Specific epoch checkpoint to load (defaults to latest).")
    parser.add_argument("--split", type=str, default="test_clean", help="Dataset split to evaluate on.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to time.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max new tokens for generation timing.")
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable KV-cache timing and use full generate timing instead.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve configuration and model initialization
    if args.experiment:
        exp_dir = Path(args.experiment)
        config_path = exp_dir / "config.yaml"
        if not config_path.exists():
            raise ValueError(f"Config file not found in experiment: {config_path}")
        cfg = ConfigManager().load_config(str(config_path))
        model_cfg = cfg["model"]
        dataset_cfg = cfg["dataset"]
        loss_cfg = cfg["loss"]
        task_cfg = cfg["task"]

        # Build task (same type/kwargs as training)
        task = create_task(task_type=task_cfg["type"], **task_cfg.get("kwargs", {}))

        # Build full model and load checkpoint
        taudio_config = {**model_cfg, **loss_cfg, "task": task}
        model = TAudio(**taudio_config).to(device)

        # Find checkpoint in experiment directory
        checkpoint_path = ConfigManager().get_model_checkpoint(exp_dir, args.epoch)
        if checkpoint_path is None:
            raise ValueError(f"No checkpoint found in {exp_dir}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        adapter = model.model_adapter
        audio_layer = model.audio_layer
        scaling_factor = model_cfg.get("scaling_factor", 1)
    else:
        # Backward compatibility: --config path without checkpoint
        if not args.config:
            raise ValueError("Either --experiment or --config must be provided.")
        cfg = ConfigManager().load_config(args.config)
        model_cfg = cfg["model"]
        dataset_cfg = cfg["dataset"]
        loss_cfg = cfg["loss"]
        task_cfg = cfg["task"]

        # Create adapter/model components (no checkpoint)
        dtype_cfg = model_cfg.get("dtype", "auto")
        if isinstance(dtype_cfg, str) and dtype_cfg.lower() == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = "auto"

        adapter = create_adapter(
            model_id=model_cfg["model_id"],
            bidirectional_audio=model_cfg.get("bidirectional_audio", False),
            dtype=torch_dtype,
            scaling_factor=model_cfg.get("scaling_factor", 1),
        )
        adapter = adapter.to(device)
        adapter.eval()

        # Create a minimal linear holder to mirror trained head shape
        class _TAudioLite(torch.nn.Module):
            def __init__(self, audio_layer: int, scaling_factor: int):
                super().__init__()
                self.model_adapter = adapter
                self.audio_layer = audio_layer
                self.linear = torch.nn.Linear(adapter.hidden_dim, scaling_factor, dtype=adapter.dtype).to(next(adapter.parameters()).device)

        model = _TAudioLite(audio_layer=model_cfg["audio_layer"], scaling_factor=model_cfg.get("scaling_factor", 1))
        model.eval()
        audio_layer = model.audio_layer
        scaling_factor = model_cfg.get("scaling_factor", 1)

    # Build dataset adapter directly (no transforms), to use task.build_labels(eval_mode=True)
    repo = dataset_cfg["repository"]
    ds_adapter = create_ds_adapter(
        infer_adapter_from_repository(repo),
        sampling_rate=adapter.sampling_rate,
        repository=repo,
        take_first=None,
        left_padding=0,
        key=task_cfg["kwargs"].get("key", "start"),
    )
    base_ds = ds_adapter.load_split(args.split)

    # Prepare tasks
    task_single = create_task("SINGLE_WORD_TIMESTAMP", **task_cfg.get("kwargs", {}))
    task_single_any = create_task("SINGLE_WORD_TIMESTAMP_ANY", **task_cfg.get("kwargs", {}))

    # Accumulators
    token_times_single: List[float] = []
    token_times_any: List[float] = []
    surrogate_times_single: List[float] = []
    surrogate_times_any: List[float] = []

    # Iterate examples
    num = min(args.num_samples, len(base_ds))
    for i in range(num):
        example = base_ds[i]
        # Build eval-mode inputs for each task (these include the audio + prompt)
        inputs_single = task_single.build_labels(
            example=example,
            ds_adapter=ds_adapter,
            model_adapter=adapter,
            eval_mode=True,
        )
        inputs_any = task_single_any.build_labels(
            example=example,
            ds_adapter=ds_adapter,
            model_adapter=adapter,
            eval_mode=True,
        )

        # Token-loss generation timing
        if args.no_kv_cache:
            t_single, _, txt_single = measure_generation_time_no_kv(
                adapter=adapter,
                processor=adapter.processor,
                inputs=inputs_single,
                max_new_tokens=args.max_new_tokens,
            )
            t_any, _, txt_any = measure_generation_time_no_kv(
                adapter=adapter,
                processor=adapter.processor,
                inputs=inputs_any,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            t_single, _, txt_single = measure_generation_time_kv(
                adapter=adapter,
                processor=adapter.processor,
                inputs=inputs_single,
                max_new_tokens=args.max_new_tokens,
            )
            t_any, _, txt_any = measure_generation_time_kv(
                adapter=adapter,
                processor=adapter.processor,
                inputs=inputs_any,
                max_new_tokens=args.max_new_tokens,
            )
        token_times_single.append(t_single)
        token_times_any.append(t_any)
        print(f"[decoded][timestamp_single]: {txt_single}")
        print(f"[decoded][timestamp_single_any]: {txt_any}")

        # Surrogate timing (linear + poisson/bernoulli inference only)
        s_single = measure_surrogate_time(
            taudio_model=model,
            task=task_single,
            adapter=adapter,
            inputs=inputs_single,
            poisson_loss=loss_cfg.get("poisson_loss", False),
            class_weighting=loss_cfg.get("class_weighting", False),
        )
        s_any = measure_surrogate_time(
            taudio_model=model,
            task=task_single_any,
            adapter=adapter,
            inputs=inputs_any,
            poisson_loss=loss_cfg.get("poisson_loss", False),
            class_weighting=loss_cfg.get("class_weighting", False),
        )
        surrogate_times_single.append(s_single)
        surrogate_times_any.append(s_any)

    def _avg(xs: List[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    print("Token generation (KV-cache decode only):")
    print(f"  timestamp_single:      avg={_avg(token_times_single):.6f}s  samples={len(token_times_single)}")
    print(f"  timestamp_single_any:  avg={_avg(token_times_any):.6f}s  samples={len(token_times_any)}")

    print("Surrogate path (linear + inference only):")
    print(f"  timestamp_single:      avg={_avg(surrogate_times_single):.6f}s  samples={len(surrogate_times_single)}")
    print(f"  timestamp_single_any:  avg={_avg(surrogate_times_any):.6f}s  samples={len(surrogate_times_any)}")


if __name__ == "__main__":
    main()


