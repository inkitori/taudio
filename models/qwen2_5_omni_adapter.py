from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from utils.utils import get_audio_bounds
from utils import qwen_3b_constants, qwen_7b_constants
from contextlib import contextmanager
import types
import logging
from models.base_model_adapter import BaseModelAdapter
from contextlib import nullcontext


class Qwen2_5OmniAdapter(BaseModelAdapter):
    def __init__(self, model_id: str, load_in_8bit: bool, bidirectional_audio: bool, dtype: str) -> None:
        super().__init__()
        self.base_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            load_in_8bit=load_in_8bit,
        )

        # Convenience references
        self._audio_tower = self.base_model.audio_tower
        self._text_model = self.base_model.model

        self._processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        self.bidirectional_audio = bidirectional_audio

        if model_id.lower() == "qwen/qwen2.5-omni-3b":
            self.constants = qwen_3b_constants
        elif model_id.lower() == "qwen/qwen2.5-omni-7b":
            self.constants = qwen_7b_constants
        else:
            raise ValueError(f"Unsupported model: {model_id}")

    def enable_gradient_checkpointing(self):
        self.base_model.gradient_checkpointing_enable()
        self._audio_tower.gradient_checkpointing_enable()
        self._text_model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing")

    # Properties required by TAudio
    @property
    def hidden_dim(self) -> int:
        return self.base_model.config.audio_config.output_dim

    @property
    def sampling_rate(self) -> int:
        return 16000

    @property
    def dtype(self) -> torch.dtype:
        return self.base_model.dtype

    @property
    def audio_id(self) -> int:
        return self.base_model.config.audio_token_index

    @property
    def assistant_id(self) -> int:
        return self.constants.ASSISTANT_ID

    @property
    def text_model(self) -> nn.Module:
        return self._text_model

    @property
    def processor(self) -> Qwen2_5OmniProcessor:
        return self._processor

    # Pass-throughs
    def forward(
        self,
        **kwargs,
    ) -> Any:
        with self.bidirectional_audio_context(kwargs["input_ids"]) if self.bidirectional_audio else nullcontext():
            logging.info(f"Using bidirectional audio context: {
                         self.bidirectional_audio}")
            return self.base_model(
                **kwargs,
            )

    def generate(self, **kwargs):
        with self.bidirectional_audio_context(kwargs["input_ids"]) if self.bidirectional_audio else nullcontext():
            logging.info(f"Using bidirectional audio context: {
                         self.bidirectional_audio}")
            return self.base_model.generate(**kwargs)

    @property
    def seconds_to_embedding(self) -> int:
        return self.constants.SECONDS_TO_EMBEDDING

    # --- Bidirectional audio mask patching (Qwen-specific) ---
    def _patch_causal_mask_zero_region(self, start: int, end: int):
        model = self._text_model
        original_method = model._update_causal_mask

        model.mask_start = start
        model.mask_end = end

        def patched_update_causal_mask(self_ref, attention_mask, input_tensor, cache_position, past_key_values, output_attentions=False):
            causal_mask = original_method(
                attention_mask, input_tensor, cache_position, past_key_values, output_attentions
            )

            if (
                causal_mask is not None
                and hasattr(self_ref, 'mask_start') and hasattr(self_ref, 'mask_end')
                and self_ref.mask_start is not None and self_ref.mask_end is not None
            ):
                start_idx = self_ref.mask_start
                end_idx = self_ref.mask_end
                if (
                    causal_mask.dim() >= 4
                    and 0 <= start_idx < causal_mask.shape[2]
                    and 0 <= end_idx < causal_mask.shape[3]
                    and start_idx <= end_idx
                ):
                    causal_mask[0, 0, start_idx:end_idx +
                                1, start_idx:end_idx + 1] = 0
                    logging.info(f"Zeroed out causal mask region [{start_idx}:{
                                 end_idx + 1}, {start_idx}:{end_idx + 1}]")

            return causal_mask

        model._update_causal_mask = types.MethodType(
            patched_update_causal_mask, model)
        logging.info(f"Patched _update_causal_mask on {type(model).__name__}")

    def _unpatch_causal_mask(self):
        model = self._text_model
        original_method = type(model)._update_causal_mask
        model._update_causal_mask = types.MethodType(original_method, model)
        if hasattr(model, 'mask_start'):
            delattr(model, 'mask_start')
        if hasattr(model, 'mask_end'):
            delattr(model, 'mask_end')
        logging.info(f"Unpatched _update_causal_mask on {
                     type(model).__name__}")

    @contextmanager
    def bidirectional_audio_context(self, input_ids: torch.Tensor):
        start_audio_index, end_audio_index = get_audio_bounds(
            input_ids, self.constants.BEGIN_AUDIO_ID, self.constants.END_AUDIO_ID)
        self._patch_causal_mask_zero_region(start_audio_index, end_audio_index)
        logging.info(f"Enabled bidirectional audio processing for region [{
                     start_audio_index}:{end_audio_index}]")
        try:
            yield
        finally:
            self._unpatch_causal_mask()
            logging.info("Restored original causal mask settings")
