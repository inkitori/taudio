import numpy as np

from typing import Any

import torch
import torch.nn as nn
from transformers import VoxtralForConditionalGeneration
from utils.utils import get_audio_bounds
from contextlib import contextmanager
import types
import logging
from models.base_model_adapter import BaseModelAdapter
from contextlib import nullcontext
# Import FSDP to check for it
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoProcessor
from utils.utils import ensure_audio_path


class VoxtralAdapter(BaseModelAdapter):
    def __init__(self, model_id: str, bidirectional_audio: bool, dtype: str, scaling_factor: int) -> None:
        super().__init__()
        self.base_model = VoxtralForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        self._processor = AutoProcessor.from_pretrained(model_id)

    @property
    def processor(self) -> AutoProcessor:
        return self._processor

    def generate(self, decode_tokens: bool = False, **kwargs):
        outputs = self.base_model.generate(**kwargs)
        if decode_tokens:
            decoded_outputs = self.processor.batch_decode(outputs[:, kwargs["input_ids"].shape[1]:], skip_special_tokens=True)
            return decoded_outputs[0]
        else:
            return outputs[:, kwargs["input_ids"].shape[1]:]

    @property
    def sampling_rate(self) -> int:
        return 16000

    @property
    def hidden_dim(self) -> int:
        return 1280

    @property
    def dtype(self) -> torch.dtype:
        return self.base_model.dtype

    @property
    def text_model(self) -> nn.Module:
        return self.base_model.model


    def build_base_inputs(self, prompt: str, audio):
        audio_path = ensure_audio_path(audio)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            # add_generation_prompt=True,
            return_tensors="pt",)

        return inputs