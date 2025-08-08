from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from utils.utils import get_audio_bounds
from utils.qwen2_5_omni_constants import BEGIN_AUDIO_ID, END_AUDIO_ID
from contextlib import contextmanager
import types
import logging
from models.base_adapter import BaseAudioTextAdapter


class Qwen2_5OmniAdapter(BaseAudioTextAdapter):
    def __init__(self, model_id: str, load_in_8bit: bool) -> None:
        super().__init__()
        self.base_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            load_in_8bit=load_in_8bit,
        )

        # Convenience references
        self._audio_tower = self.base_model.audio_tower
        self._text_model = self.base_model.model

    # Properties required by TAudio
    @property
    def hidden_dim(self) -> int:
        return self.base_model.config.audio_config.output_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.base_model.dtype

    @property
    def audio_token_index(self) -> int:
        return self.base_model.config.audio_token_index

    @property
    def text_model(self) -> nn.Module:
        return self._text_model

    # Pass-throughs
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        labels: torch.Tensor,
        output_hidden_states: bool,
    ) -> Any:
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            output_hidden_states=output_hidden_states,
            labels=labels,
        )

    def generate(self, **kwargs):
        return self.base_model.generate(**kwargs)

    def get_audio_bounds(self, input_ids: torch.Tensor) -> tuple[int, int]:
        return get_audio_bounds(input_ids, BEGIN_AUDIO_ID, END_AUDIO_ID)

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
        start_audio_index, end_audio_index = self.get_audio_bounds(input_ids)
        self._patch_causal_mask_zero_region(start_audio_index, end_audio_index)
        logging.info(f"Enabled bidirectional audio processing for region [{
                     start_audio_index}:{end_audio_index}]")
        try:
            yield
        finally:
            self._unpatch_causal_mask()
            logging.info("Restored original causal mask settings")
