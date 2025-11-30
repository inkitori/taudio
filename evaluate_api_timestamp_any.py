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
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import numpy as np
import soundfile as sf
import wandb

from dataset import create_adapter as create_dataset_adapter
from dataset import infer_adapter_from_repository
from tasks.timestamp_single_any import SingleTimestampAnyTask
from utils.metrics import AverageMetrics
from utils.utils import round_timestamp_python, ensure_audio_path


DATASET_REPOS: Dict[str, str] = {
    "librispeech": "gilkeyio/librispeech-alignments",
    "libricount": "enyoukai/libricount-timings",
    "audioset": "enyoukai/AudioSet-Strong-Human-Sounds",
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
    def __init__(self, api_key: str, model: str, timeout: Optional[int]) -> None:
        self.name = "gemini"
        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency hint
            raise RuntimeError(
                "google-genai is required. Install via `pip install google-genai`."
            ) from exc

        if not api_key:
            raise ValueError("Missing Gemini API key. Set --gemini-api-key or GEMINI_API_KEY.")

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._timeout = timeout
        self._gen_config = {"temperature": 0.0, "max_output_tokens": 256}

    def predict_timestamp(self, prompt: str, audio: Dict[str, object]) -> str:
        audio_path = ensure_audio_path(audio)
        uploaded_file = None
        try:
            uploaded_file = self._client.files.upload(
                file=audio_path,
            )
            request_kwargs = {"config": self._gen_config}
            if self._timeout:
                request_kwargs["request_options"] = {"timeout": self._timeout}

            contents = [prompt, uploaded_file]
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                **request_kwargs,
            )
            return _extract_gemini_text(response)
        finally:
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
            # `timeout` lives inside request options for the new SDK.
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


def _extract_gemini_text(response: object) -> str:
    text = getattr(response, "text", "")
    if text:
        return str(text).strip()

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""

    texts = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", []) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                texts.append(str(part_text))
    return "\n".join(texts).strip()


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
        f"{base_prompt} Respond with only the timestamp in seconds using a decimal number such as '2.435'."
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

    processed = 0
    for example in dataset:
        if task.skip_example(example, ds_adapter):
            continue
        try:
            prompt, audio, gt = _build_prompt(ds_adapter, task, example)
        except ValueError:
            continue

        logging.info(f"Prompt: " + prompt)
        raw_text = predictor.predict_timestamp(prompt, audio)
        logging.info(
            "[%s][%s] Model response: %s | Ground truth: %.3f",
            predictor.name,
            repository,
            raw_text,
            gt,
        )
        parsed = _parse_timestamp(raw_text)
        if parsed is None:
            parsed = _fallback_timestamp(audio)
        
        logging.info(f"Parsed timestamp: " + str(parsed))
        example_metrics = _metrics_from_prediction(parsed, gt)
        logging.info(f"Current metrics: ")
        logging.info(example_metrics)
        metrics.update_dict(example_metrics)
        processed += 1

        if wandb_run:
            wandb_run.log(metrics.to_dict())

        if max_examples and processed >= max_examples:
            break

    return EvaluationResult(processed=processed, metrics=metrics.to_dict())


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

    return parser.parse_args()


def build_predictor(name: str, args: argparse.Namespace) -> TimestampPredictor:
    timeout = args.request_timeout
    if name == "gemini":
        return GeminiTimestampClient(args.gemini_api_key, args.gemini_model, timeout)
    if name == "chatgpt":
        return ChatGPTEvaluator(args.chatgpt_api_key, args.chatgpt_model, timeout)
    raise ValueError(f"Unsupported provider: {name}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    _silence_third_party_logs()

    random.seed(args.seed)
    np.random.seed(args.seed)

    task = SingleTimestampAnyTask(key="start")

    for provider_name in args.providers:
        predictor = build_predictor(provider_name, args)
        for dataset_key in args.datasets:
            repository = DATASET_REPOS[dataset_key]
            run = None
            if args.log_wandb:
                run = wandb.init(
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    name=f"[{provider_name}][{repository}][{args.split}]",
                    config={
                        "provider": provider_name,
                        "repository": repository,
                        "split": args.split,
                        "task": "SINGLE_WORD_TIMESTAMP_ANY",
                        "max_examples": args.max_examples,
                        "model": args.gemini_model if provider_name == "gemini" else args.chatgpt_model,
                    },
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