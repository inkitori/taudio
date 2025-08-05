import torch
import torch.nn as nn
from contextlib import contextmanager, nullcontext
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from utils.causal_mask_patch import patch_causal_mask_zero_region, unpatch_causal_mask

from collections import namedtuple
from typing import Optional

from utils.qwen2_5_omni_constants import BEGIN_AUDIO_ID, END_AUDIO_ID
from utils.utils import get_audio_bounds
from utils.poisson import poisson_loss, poisson_count_loss
import logging
from dataset import TaskType

Output = namedtuple(
    'Output', ['loss', 'token_loss', 'surrogate_loss', 'pred', 'gt'])


class TAudio(nn.Module):
    def __init__(
            self,
            model_id: str,
            freeze_text_model: bool,
            load_in_8bit: bool,
            audio_layer: int,
            class_weighting: bool,
            surrogate_loss: bool,
            token_loss: bool,
            surrogate_loss_weight: float,
            bidirectional_audio: bool = False,
            poisson_loss: bool = False,
            linear_bias: Optional[float] = None,
            task_type: TaskType = TaskType.SINGLE_WORD_TIMESTAMP
    ) -> None:
        super(TAudio, self).__init__()

        self.base_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            load_in_8bit=load_in_8bit,
        )
        self.base_audio_encoder = self.base_model.audio_tower
        self.base_text_model = self.base_model.model

        self.hidden_dim = self.base_model.config.audio_config.output_dim

        if freeze_text_model:
            for param in self.base_text_model.parameters():
                param.requires_grad = False

        self.linear = nn.Linear(
            self.hidden_dim, 1, dtype=self.base_model.dtype)

        if linear_bias is not None:
            with torch.no_grad():
                logging.info(f"Filling linear bias with {linear_bias}")
                self.linear.bias.fill_(linear_bias)

        self.audio_layer = audio_layer

        self.class_weighting = class_weighting
        self.bidirectional_audio = bidirectional_audio

        self.surrogate_loss = surrogate_loss
        self.token_loss = token_loss
        self.surrogate_loss_weight = surrogate_loss_weight
        self.poisson_loss = poisson_loss
        self.task_type = task_type

    def get_audio_token_id(self) -> int:
        return self.base_model.config.audio_token_index

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch_size, seq_len)
        attention_mask: torch.Tensor,  # (batch_size, seq_len)
        # (batch_size, embedding_dim, audio_context_len)
        input_features: torch.Tensor,
        # (batch_size, audio_context_len)
        feature_attention_mask: torch.Tensor,
        labels: torch.Tensor,  # (num_audio_tokens)
        label_ids: torch.Tensor  # (batch_size, seq_len)
    ) -> torch.Tensor:
        with self.bidirectional_audio_context(input_ids) if self.bidirectional_audio else nullcontext():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                output_hidden_states=True,
                labels=label_ids,
            )

        # (batch_size, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states[self.audio_layer]

        # (num_audio_tokens, hidden_dim)
        audio_hidden_states = hidden_states[input_ids ==
                                            self.get_audio_token_id()]

        # (num_audio_tokens across batch)
        logits = self.linear(audio_hidden_states).squeeze()
        labels = labels.to(logits.dtype)

        # TODO: fix this code for batched inputs
        pred_top_val, pred_top_idx = torch.max(logits, dim=0)
        gt_top_val, gt_top_idx = torch.max(labels, dim=0)

        if self.surrogate_loss:
            surrogate_loss = self.surrogate_loss_handler(logits, labels)
        else:
            surrogate_loss = torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype)

        if self.token_loss:
            token_loss = outputs.loss
        else:
            token_loss = torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype)

        loss = token_loss + self.surrogate_loss_weight * surrogate_loss

        output = Output(loss=loss, surrogate_loss=surrogate_loss, token_loss=token_loss, pred=(
            pred_top_val, pred_top_idx), gt=(gt_top_val, gt_top_idx))

        return output

    def generate(self, **kwargs):
        with self.bidirectional_audio_context(kwargs['input_ids']) if self.bidirectional_audio else nullcontext():
            return self.base_model.generate(**kwargs)

    @contextmanager
    def bidirectional_audio_context(self, input_ids: torch.Tensor):
        """
        Context manager that temporarily patches the causal mask to enable bidirectional audio processing
        for the specified audio token range.

        Args:
            start_audio_index: Start index for the audio region to make bidirectional
            end_audio_index: End index for the audio region to make bidirectional

        Usage:
            with model.bidirectional_audio(start_idx, end_idx):
                # Code that runs with bidirectional audio processing
                outputs = model(...)
        """
        try:
            start_audio_index, end_audio_index = get_audio_bounds(
                input_ids, BEGIN_AUDIO_ID, END_AUDIO_ID)
            patch_causal_mask_zero_region(
                self.base_text_model, start_audio_index, end_audio_index)

            logging.debug(f"Start: {start_audio_index}, End: {
                          end_audio_index}")
            logging.debug(
                f"Patched: {input_ids[0, start_audio_index:end_audio_index + 1]}")
            logging.debug(f"Unpatched: {input_ids[0, :start_audio_index]} {
                          input_ids[0, end_audio_index + 1:]}")

            logging.info(f"Enabled bidirectional audio processing for region [{
                         start_audio_index}:{end_audio_index}]")

            yield

        finally:
            unpatch_causal_mask(self.base_text_model)
            logging.info(f"Restored original causal mask settings")

    def surrogate_loss_handler(self, logits, labels):
        if self.task_type == TaskType.SINGLE_WORD_TIMESTAMP:
            return self.single_word_timestamp_surrogate_loss(logits, labels)
        elif self.task_type == TaskType.SPEAKER_COUNTING:
            return self.speaker_counting_surrogate_loss(logits, labels)
        else:
            raise ValueError(f"Task type {self.task_type} not supported")

    def single_word_timestamp_surrogate_loss(self, logits, labels):
        if self.poisson_loss:
            return poisson_loss(logits.unsqueeze(
                0), labels.unsqueeze(0), torch.ones_like(logits.unsqueeze(0)))
        else:
            if self.class_weighting:
                num_ones = (labels == 1).sum()
                num_zeros = (labels == 0).sum()
                pos_weight = (
                    num_zeros / num_ones) if num_ones > 0 else 1.0

                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = nn.BCEWithLogitsLoss()

            return criterion(logits, labels)

    def speaker_counting_surrogate_loss(self, logits, labels):
        if self.poisson_loss:
            ground_truth_count = labels.sum()
            return poisson_count_loss(logits.unsqueeze(
                0), ground_truth_count.unsqueeze(0), torch.ones_like(logits.unsqueeze(0)))
        else:
            raise ValueError(
                "Only Poisson loss is supported for speaker counting")
