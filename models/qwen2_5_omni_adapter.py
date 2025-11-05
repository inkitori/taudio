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
# Import FSDP to check for it
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from utils.utils import ensure_audio_path


class Qwen2_5OmniAdapter(BaseModelAdapter):
    def __init__(self, model_id: str, bidirectional_audio: bool, dtype: str, scaling_factor: int) -> None:
        super().__init__()
        self.base_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        # Convenience references
        self._processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        self.bidirectional_audio = bidirectional_audio
        self.scaling_factor = scaling_factor

        if model_id.lower() == "qwen/qwen2.5-omni-3b":
            self.constants = qwen_3b_constants
        elif model_id.lower() == "qwen/qwen2.5-omni-7b":
            self.constants = qwen_7b_constants
        else:
            raise ValueError(f"Unsupported model: {model_id}")

    # --- START: NEW HELPER PROPERTY ---
    @property
    def _actual_text_model(self) -> nn.Module:
        """
        Returns the underlying text model, handling the case where it might be
        wrapped by FSDP.
        """
        if isinstance(self.base_model.model, FSDP):
            # If it's an FSDP instance, the real module is in the .module attribute
            return self.base_model.model.module
        else:
            # Otherwise, return it directly
            return self.base_model.model
    # --- END: NEW HELPER PROPERTY ---

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
        # This property should still return the FSDP wrapper if it exists,
        # as the training loop needs to interact with it.
        return self.base_model.model

    @property
    def processor(self) -> Qwen2_5OmniProcessor:
        return self._processor

    # Pass-throughs
    def forward(
        self,
        **kwargs,
    ) -> Any:
        with self.bidirectional_audio_context(kwargs["input_ids"]) if self.bidirectional_audio else nullcontext():
            logging.debug(f"Using bidirectional audio context: {
                         self.bidirectional_audio}")
            return self.base_model(
                **kwargs,
            )

    def generate(self, decode_tokens: bool = False, **kwargs):
        with self.bidirectional_audio_context(kwargs["input_ids"]) if self.bidirectional_audio else nullcontext():
            logging.debug(f"Using bidirectional audio context: {
                         self.bidirectional_audio}")

            tokens = self.base_model.generate(**kwargs)

            if decode_tokens:
                generated_tokens = tokens[0][kwargs["input_ids"].shape[1]:-1]
                generated_string = self.processor.tokenizer.decode(generated_tokens)
                return generated_string
            else:
                return tokens

    @property
    def seconds_to_embedding(self) -> int:
        return self.constants.SECONDS_TO_EMBEDDING

    # --- Bidirectional audio mask patching (Qwen-specific) ---
    def _patch_causal_mask_zero_region(self, starts, ends):
        model = self._actual_text_model
        # This is the original method from the Qwen2_5OmniThinkerTextModel class
        original_method = model._update_causal_mask

        # Persist per-batch start/end indices on the model
        model.mask_starts = starts
        model.mask_ends = ends

        def patched_update_causal_mask(self_ref, attention_mask, input_tensor, cache_position, past_key_values, output_attentions=False):
            # First, call the original method to get the base causal mask
            causal_mask = original_method(
                attention_mask, input_tensor, cache_position, past_key_values, output_attentions
            )
            
            logging.debug(f"Causal mask: {causal_mask.shape}")
            logging.debug(f"Input tensor: {input_tensor.shape}")

            if (
                causal_mask is not None
                and hasattr(self_ref, 'mask_starts') and hasattr(self_ref, 'mask_ends')
                and self_ref.mask_starts is not None and self_ref.mask_ends is not None
            ):
                # Normalize to list for uniform handling
                mask_starts = self_ref.mask_starts
                mask_ends = self_ref.mask_ends
                if not isinstance(mask_starts, (list, tuple)):
                    mask_starts = [int(mask_starts)] * causal_mask.shape[0]
                if not isinstance(mask_ends, (list, tuple)):
                    mask_ends = [int(mask_ends)] * causal_mask.shape[0]

                # Clone to safely modify
                cloned_causal_mask = causal_mask.clone()
                batch_size = cloned_causal_mask.shape[0]

                for b in range(batch_size):
                    start_idx = int(mask_starts[b])
                    end_idx = int(mask_ends[b])
                    if (
                        0 <= start_idx < cloned_causal_mask.shape[2]
                        and 0 <= end_idx < cloned_causal_mask.shape[3]
                        and start_idx <= end_idx
                    ):
                        # Zero region for all heads in this batch element
                        cloned_causal_mask[b, :, start_idx:end_idx + 1, start_idx:end_idx + 1] = 0
                    else:
                        logging.debug(f"Start index {start_idx} or end index {end_idx} is out of range for batch {b}")

                return cloned_causal_mask

            # If no modifications were made, return the original mask
            return causal_mask

        model._update_causal_mask = types.MethodType(
            patched_update_causal_mask, model)
        logging.debug(f"Patched _update_causal_mask on {type(model).__name__}")

    def _unpatch_causal_mask(self):
        # Use the new helper property to get the actual model
        model = self._actual_text_model
        original_method = type(model)._update_causal_mask
        model._update_causal_mask = types.MethodType(original_method, model)
        if hasattr(model, 'mask_starts'):
            delattr(model, 'mask_starts')
        if hasattr(model, 'mask_ends'):
            delattr(model, 'mask_ends')
        logging.debug(f"Unpatched _update_causal_mask on {
                     type(model).__name__}")

    @contextmanager
    def bidirectional_audio_context(self, input_ids: torch.Tensor):
        starts, ends = get_audio_bounds(
            input_ids, self.constants.BEGIN_AUDIO_ID, self.constants.END_AUDIO_ID)
        self._patch_causal_mask_zero_region(starts, ends)
        logging.debug("Enabled bidirectional audio processing for region(s)")
        try:
            yield
        finally:
            self._unpatch_causal_mask()
            logging.debug("Restored original causal mask settings")
    
    def build_base_inputs(self, prompt: str, audio):
        # audio_path = ensure_audio_path(audio)
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "audio": "PLACEHOLDER"},
                ],
            },
        ]

        conversation_template = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=conversation_template,
            audio=audio["array"],
            return_tensors="pt",
            padding=True,
        )

        return inputs