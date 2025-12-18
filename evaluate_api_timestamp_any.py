#!/usr/bin/env python3
"""
Evaluate timestamp_single_any on LibriSpeech / LibriCount / AudioSet using
Gemini and ChatGPT APIs.

This script mirrors `evaluate_base.py` but swaps the local Hugging Face models
for hosted APIs so we can establish baseline quality for off-the-shelf speech
models.  Each provider receives the same prompt/audio pair and we reuse the
existing dataset/task adapters to stay consistent with the rest of the codebase.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple, List, Any
from pydantic import BaseModel, Field

import numpy as np
import soundfile as sf
import wandb

from dataset import create_adapter as create_dataset_adapter
from dataset import infer_adapter_from_repository
from tasks.timestamp_single_any import SingleTimestampAnyTask
from utils.metrics import AverageMetrics
from utils.utils import round_timestamp_python, ensure_audio_path

try:
    from google import genai  # type: ignore
    from google.genai import types
except ImportError as exc:  # pragma: no cover - dependency hint
    raise RuntimeError(
        "google-genai is required. Install via `pip install google-genai`."
    ) from exc

class Timestamp(BaseModel):
    # event_name: str = Field(description="Name of the event")
    start: float = Field(description="Start time of the event")

class Timestamps(BaseModel):
    timestamps: List[Timestamp]

DATASET_REPOS: Dict[str, str] = {
    "librispeech": "gilkeyio/librispeech-alignments",
    "libricount": "enyoukai/libricount-timings",
    "audioset": "enyoukai/audioset-humans-reprocessed",
}

FLOAT_PATTERN = re.compile(r"\d+(?:\.\d+)?")
TIMECODE_PATTERN = re.compile(r"(\d+):(\d+(?:\.\d+)?)")
THIRD_PARTY_LOGGERS = (
    "httpx",
    "google",
    "google.api_core",
    "google.auth",
    "google.cloud",
    "google.genai",
    "google.generativeai",
    "google_genai",
    "google_genai.models",
)
def _silence_third_party_logs(level: int = logging.WARNING) -> None:
    """Silence noisy SDK loggers without touching application logging."""
    for name in THIRD_PARTY_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())


class TimestampPredictor(Protocol):
    """Lightweight interface for hosted speech models."""

    name: str

    def predict_timestamp(self, prompt: str, audio: Dict[str, object]) -> str:
        """Return the raw text response from the provider."""


class GeminiTimestampClient:
    def __init__(self, api_key: str, model: str, timeout: Optional[int], max_retries: int = 5) -> None:
        self.name = "gemini"
        self._max_retries = max_retries

        if not api_key:
            raise ValueError("Missing Gemini API key. Set --gemini-api-key or GEMINI_API_KEY.")

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._timeout = timeout

    def predict_timestamp(self, prompt: str, audio: Dict[str, object]) -> str:
        audio_path = ensure_audio_path(audio)
        
        # Exponential backoff loop
        for attempt in range(self._max_retries):
            uploaded_file = None
            try:
                # We re-upload on every retry to ensure clean state, 
                # as failed requests might leave file states ambiguous on the server side 
                # or the file handle might be consumed.
                uploaded_file = self._client.files.upload(
                    file=audio_path,
                )
                
                contents = [prompt, uploaded_file]
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_json_schema=Timestamp.model_json_schema(),
                    ),
                )
                
                logging.info(f"[GEMINI] Response: {response}")
                logging.info(f"[GEMINI] Response Text: {response.text}")
                
                # If successful, validate and return
                result = Timestamp.model_validate_json(response.text).model_dump()
                
                # Clean up success case
                self._cleanup(uploaded_file, audio_path)
                return result

            except Exception as e:
                # Log the error
                logging.warning(f"[GEMINI] Attempt {attempt + 1}/{self._max_retries} failed: {e}")
                
                # Cleanup temporary upload if it exists before retrying
                if uploaded_file:
                    try:
                        self._client.files.delete(name=uploaded_file.name)
                    except Exception:
                        logging.warning(f"[GEMINI] Failed to delete files on client")

                # Check if we should retry or raise
                if attempt == self._max_retries - 1:
                    # Final attempt failed, clean up local file and raise
                    try:
                        os.remove(audio_path)
                    except OSError:
                        logging.warning(f"[GEMINI] Failed to delete files locally")
                    logging.error(f"[GEMINI] All {self._max_retries} attempts failed.")
                    raise e
                
                # Exponential backoff: 2, 4, 8, 16...
                sleep_time = 2 ** (attempt + 1)
                logging.info(f"[GEMINI] Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

        return "" # Should not be reached due to raise above

    def _cleanup(self, uploaded_file, audio_path):
        try:
            os.remove(audio_path)
        except OSError:
            logging.warning("Could not remove temporary audio file %s", audio_path)
        if uploaded_file:
            try:
                self._client.files.delete(name=uploaded_file.name)
            except Exception:
                logging.warning("Failed to delete Gemini uploaded file %s", uploaded_file.name)


class ChatGPTEvaluator:
    def __init__(self, api_key: str, model: str, timeout: Optional[int]) -> None:
        self.name = "chatgpt"
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency hint
            raise RuntimeError(
                "openai>=1.0 is required. Install via `pip install openai`."
            ) from exc

        if not api_key:
            raise ValueError("Missing OpenAI API key. Set --chatgpt-api-key or OPENAI_API_KEY.")

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._timeout = timeout

    def predict_timestamp(self, prompt: str, audio: Dict[str, object]) -> str:
        # Note: Depending on flakiness, retry logic could be added here too, 
        # but the request specifically asked for Gemini.
        audio_bytes = _audio_to_wav_bytes(audio)
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
        response = self._client.responses.create(
            model=self._model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_audio", "audio": {"data": encoded_audio, "format": "wav"}},
                    ],
                }
            ],
            max_output_tokens=64,
            temperature=0.0,
            top_p=0.1,
            **({"timeout": self._timeout} if self._timeout else {}),
        )
        return _extract_openai_text(response)


def _audio_to_wav_bytes(audio: Dict[str, object]) -> bytes:
    buffer = io.BytesIO()
    samples = np.asarray(audio["array"], dtype=np.float32)
    sr = int(audio["sampling_rate"])
    sf.write(buffer, samples, sr, format="WAV")
    buffer.seek(0)
    return buffer.read()


def _extract_openai_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    output = getattr(response, "output", None)
    if not output:
        return ""

    texts = []
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []):
            if getattr(content, "type", None) in {"output_text", "text"}:
                texts.append(str(getattr(content, "text", "")))
    return "\n".join(texts).strip()


def _build_prompt(ds_adapter, task: SingleTimestampAnyTask, example: Dict[str, object]) -> Tuple[str, Dict[str, object], float]:
    events = list(ds_adapter.get_events(example))
    event = task._choose_event(events=events, ds_adapter=ds_adapter, apply_fallback=False)  # type: ignore[attr-defined]
    event_name = ds_adapter.event_name(event)
    ordinal = task._compute_ordinal(all_events=events, ds_adapter=ds_adapter, selected_event=event)
    base_prompt = ds_adapter.get_timestamp_single_any_prompt(event_name, task.key, ordinal)
    prompt = (
        f"{base_prompt}"
    )
    audio = ds_adapter.get_audio(example)
    gt = ds_adapter.get_target_seconds(event, task.key)
    return prompt, audio, gt


def _fallback_timestamp(audio: Dict[str, object]) -> float:
    samples = np.asarray(audio["array"])
    sr = int(audio["sampling_rate"])
    return float(samples.size / (2 * sr))


def _parse_timestamp(raw_text: str) -> Optional[float]:
    if not raw_text:
        return None

    timecode_match = TIMECODE_PATTERN.search(raw_text)
    if timecode_match:
        minutes = float(timecode_match.group(1))
        seconds = float(timecode_match.group(2))
        return minutes * 60 + seconds

    matches = FLOAT_PATTERN.findall(raw_text)
    if matches:
        return float(matches[-1])
    return None

def _parse_timestamp_json(response_json) -> Optional[float]:
    if not response_json:
        return None
    try:
        timestamp = response_json['start']
    except:
        timestamp = None
    return timestamp


def _metrics_from_prediction(pred: float, gt: float) -> Dict[str, float]:
    pred = round_timestamp_python(float(pred))
    gt = round_timestamp_python(float(gt))
    abs_err = round_timestamp_python(abs(pred - gt))
    thresholds = [0.005, 0.010, 0.020, 0.040, 0.050, 0.080, 0.100, 0.200]
    metric_dict = {"token_abs_error_sum": abs_err}
    for threshold in thresholds:
        key = f"token_correct_{int(threshold * 1000)}ms"
        metric_dict[key] = 1.0 if abs_err <= threshold else 0.0
    return metric_dict


@dataclass
class EvaluationResult:
    processed: int
    metrics: Dict[str, float]


def evaluate_dataset(
    *,
    repository: str,
    split: str,
    task: SingleTimestampAnyTask,
    predictor: TimestampPredictor,
    sampling_rate: int,
    max_examples: Optional[int],
    wandb_run,
    left_padding: float,
    resume_from_count: int = 0,
    initial_metrics: Optional[Dict[str, float]] = None
) -> EvaluationResult:
    dataset_name = infer_adapter_from_repository(repository)
    ds_adapter = create_dataset_adapter(
        dataset_name,
        sampling_rate=sampling_rate,
        repository=repository,
        left_padding=left_padding,
        key=task.key,
    )
    dataset = ds_adapter.load_split(split)
    
    metrics = AverageMetrics()
    # Restore metrics if we are resuming
    if resume_from_count > 0 and initial_metrics:
        logging.info(f"Restoring metrics state from {resume_from_count} previous examples...")
        metrics.restore_from_averages(resume_from_count, initial_metrics)

    processed = 0
    total_processed_history = resume_from_count

    for i, example in enumerate(dataset):
        # Skip examples that were already processed in the previous run
        if i < resume_from_count:
            continue

        if task.skip_example(example, ds_adapter):
            continue
        
        try:
            prompt, audio, gt = _build_prompt(ds_adapter, task, example)
        except ValueError:
            continue

        logging.info(f"Prompt: " + prompt)
        
        try:
            raw_text = predictor.predict_timestamp(prompt, audio)
        except Exception as e:
            logging.error(f"Prediction failed permanently for example {i}: {e}")
            # Depending on policy, we might want to break or continue. 
            # Here we continue but treat as failure/fallback or just stop the run.
            # To allow resume next time, crashing might be better if it's a system issue.
            raise e

        logging.info(
            "[%s][%s] Model response: %s | Ground truth: %.3f",
            predictor.name,
            repository,
            raw_text,
            gt,
        )
        
        # Determine format (dict or str)
        if isinstance(raw_text, dict):
             parsed = _parse_timestamp_json(raw_text)
        else:
             parsed = _parse_timestamp(raw_text)

        if parsed is None:
            parsed = _fallback_timestamp(audio)
            logging.info(
                "[%s][%s] Failed to parse %s",
                predictor.name,
                repository,
                raw_text,
            )
        
        logging.info(f"Parsed timestamp: " + str(parsed))
        example_metrics = _metrics_from_prediction(parsed, gt)
        logging.info(f"Current metrics: ")
        logging.info(example_metrics)
        metrics.update_dict(example_metrics)
        
        processed += 1
        total_processed_history += 1

        if wandb_run:
            wandb_run.log(metrics.to_dict())

        if max_examples and total_processed_history >= max_examples:
            logging.info("Reached max examples limit.")
            break

    return EvaluationResult(processed=total_processed_history, metrics=metrics.to_dict())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate timestamp_single_any via Gemini/ChatGPT APIs.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate on.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_REPOS.keys()),
        default=list(DATASET_REPOS.keys()),
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["gemini", "chatgpt"],
        default=["gemini", "chatgpt"],
        help="Which providers to query.",
    )
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum evaluated examples per dataset.")
    parser.add_argument("--seed", type=int, default=80, help="Random seed for event selection.")
    parser.add_argument("--sampling-rate", type=int, default=16000, help="Target sampling rate for audio casting.")
    parser.add_argument("--left-padding", type=float, default=0.0, help="Seconds of left padding to mirror training setup.")

    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--gemini-api-key", default=os.environ.get("GEMINI_API_KEY"))
    parser.add_argument("--chatgpt-model", default="gpt-4o-mini-transcribe")
    parser.add_argument(
        "--chatgpt-api-key",
        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("CHATGPT_API_KEY"),
    )
    parser.add_argument("--request-timeout", type=int, default=None, help="Timeout (seconds) for API calls.")

    parser.add_argument("--log-wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb-entity", default="taudio")
    parser.add_argument("--wandb-project", default="Base Evaluations (API)")
    
    # New argument for resumption
    parser.add_argument("--resume-wandb-run-id", type=str, default=None, 
                        help="The Wandb Run ID to resume from. Must exist in the specified project/entity.")

    return parser.parse_args()


def build_predictor(name: str, args: argparse.Namespace) -> TimestampPredictor:
    timeout = args.request_timeout
    if name == "gemini":
        # Pass retry configuration here if needed, defaults are in class
        return GeminiTimestampClient(args.gemini_api_key, args.gemini_model, timeout)
    if name == "chatgpt":
        return ChatGPTEvaluator(args.chatgpt_api_key, args.chatgpt_model, timeout)
    raise ValueError(f"Unsupported provider: {name}")


def get_wandb_run_state(run_path: str) -> Tuple[int, Dict[str, float]]:
    """
    Connects to WandB API, fetches the run history, finds the last step,
    and returns the count (processed examples) and the metric dictionary.
    """
    api = wandb.Api()
    try:
        run = api.run(run_path)
    except Exception as e:
        raise ValueError(f"Could not find Wandb run at {run_path}: {e}")

    # We assume 'processed' count correlates to the number of rows in history
    # or a specific step counter if one is logged. 
    # Since the script logs every step, history length is a good proxy for 'processed' count.
    
    # Scan history is more efficient for large runs
    history = list(run.scan_history())
    if not history:
        return 0, {}

    last_row = history[-1]
    
    # Filter out system metrics (starting with _)
    metrics = {k: v for k, v in last_row.items() if not k.startswith("_")}
    
    # The script doesn't explicitly log a "step" counter variable other than relying on 
    # wandb's internal step.
    # However, since we log once per example, len(history) is the number of examples processed.
    count = len(list(run.scan_history(keys=["token_abs_error_sum"])))
    
    return count, metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    _silence_third_party_logs()

    random.seed(args.seed)
    np.random.seed(args.seed)

    task = SingleTimestampAnyTask(key="start")
    
    resume_id = args.resume_wandb_run_id
    
    # If resuming, we enforce specific logic (likely only one dataset/provider context supported per ID)
    # The script structure iterates providers then datasets. 
    # If resuming, we need to know WHICH combination we were in.
    # Simplification: If resuming, assume the args match the previous run configuration.

    for provider_name in args.providers:
        predictor = build_predictor(provider_name, args)
        for dataset_key in args.datasets:
            repository = DATASET_REPOS[dataset_key]
            
            run = None
            resume_count = 0
            initial_metrics = None
            
            # Setup Wandb
            if args.log_wandb:
                run_id = None
                resume_mode = None
                
                # Check if this specific combo matches the requested resume ID
                # (This logic assumes the user runs the script for the specific failed combo
                # or the script generates unique IDs. Here we allow manual ID override).
                if resume_id:
                    logging.info(f"Attempting to resume Wandb Run ID: {resume_id}")
                    
                    # 1. Fetch previous state via API
                    run_path = f"{args.wandb_entity}/{args.wandb_project}/{resume_id}"
                    try:
                        resume_count, initial_metrics = get_wandb_run_state(run_path)
                        logging.info(f"Recovered state: {resume_count} examples processed.")
                        logging.info(f"Recovered metrics: {initial_metrics}")
                        run_id = resume_id
                        resume_mode = "allow" # "must" or "allow"
                    except Exception as e:
                        logging.error(f"Failed to recover run state: {e}")
                        return 

                # Initialize Run
                run = wandb.init(
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    name=f"[{provider_name}][{repository}][{args.split}]" if not resume_id else None,
                    id=run_id,
                    resume=resume_mode,
                    config={
                        "provider": provider_name,
                        "repository": repository,
                        "split": args.split,
                        "task": "SINGLE_WORD_TIMESTAMP_ANY",
                        "max_examples": args.max_examples,
                        "model": args.gemini_model if provider_name == "gemini" else args.chatgpt_model,
                    } if not resume_id else None, # Don't overwrite config on resume
                )

            logging.info("Evaluating %s on %s (%s split)", provider_name, repository, args.split)
            
            result = evaluate_dataset(
                repository=repository,
                split=args.split,
                task=task,
                predictor=predictor,
                sampling_rate=args.sampling_rate,
                max_examples=args.max_examples,
                wandb_run=run,
                left_padding=args.left_padding,
                resume_from_count=resume_count,
                initial_metrics=initial_metrics
            )

            logging.info(
                "[%s][%s] Processed %d examples | Metrics: %s",
                provider_name,
                repository,
                result.processed,
                result.metrics,
            )
            if run:
                run.log(result.metrics)
                run.finish()


if __name__ == "__main__":
    main()
