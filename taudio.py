from collections import namedtuple
from typing import Optional

import torch
import torch.nn as nn

from contextlib import nullcontext

import logging
from models import create_adapter
from tasks.base import BaseTask

Output = namedtuple(
    'Output', ['loss', 'token_loss', 'surrogate_loss', 'auxiliary_deviation'])


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
        task: BaseTask,
        bidirectional_audio: bool,
        poisson_loss: bool,
        linear_bias: Optional[float],
        dtype: str,
        gradient_checkpointing: bool = False,
    ) -> None:
        super(TAudio, self).__init__()

        # Adapter-based backend
        self.adapter = create_adapter(
            model_id=model_id, load_in_8bit=load_in_8bit, bidirectional_audio=bidirectional_audio, dtype=dtype)

        self.hidden_dim = self.adapter.hidden_dim

        if freeze_text_model:
            for param in self.adapter.text_model.parameters():
                param.requires_grad = False

        self.linear = nn.Linear(self.hidden_dim, 1, dtype=self.adapter.dtype)

        if linear_bias is not None:
            with torch.no_grad():
                logging.info(f"Filling linear bias with {linear_bias}")
                self.linear.bias.fill_(linear_bias)

        self.audio_layer = audio_layer

        self.class_weighting = class_weighting

        self.surrogate_loss = surrogate_loss
        self.token_loss = token_loss
        self.surrogate_loss_weight = surrogate_loss_weight
        self.poisson_loss = poisson_loss
        self.task = task

        if gradient_checkpointing:
            self.adapter.enable_gradient_checkpointing()
            logging.info("Enabled gradient checkpointing")

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
        if labels.ndim == 2:
            labels = labels.squeeze(0)

        outputs = self.adapter(
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
                                            self.adapter.audio_id]

        # (num_audio_tokens across batch)
        logits = self.linear(audio_hidden_states).squeeze()
        labels = labels.to(logits.dtype)

        if self.surrogate_loss:
            surrogate_loss, auxiliary_deviation = self.task.calculate_loss(
                logits=logits, labels=labels, adapter=self.adapter, use_poisson_loss=self.poisson_loss, class_weighting=self.class_weighting)
        else:
            surrogate_loss = torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype)
            auxiliary_deviation = torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype)

        if self.token_loss:
            token_loss = outputs.loss
        else:
            token_loss = torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype)

        loss = token_loss + self.surrogate_loss_weight * surrogate_loss

        output = Output(loss=loss, surrogate_loss=surrogate_loss, token_loss=token_loss, auxiliary_deviation=auxiliary_deviation)

        return output

    def generate(self, **kwargs):
        return self.adapter.generate(**kwargs)
