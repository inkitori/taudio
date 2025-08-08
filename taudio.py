import torch
import torch.nn as nn
from contextlib import contextmanager, nullcontext

from collections import namedtuple
from typing import Optional

from utils.poisson import poisson_loss, poisson_count_loss
import logging
from dataset import TaskType
from models import create_adapter

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
        task_type: TaskType = TaskType.SINGLE_WORD_TIMESTAMP,
        backend: str = "qwen2_5_omni",
    ) -> None:
        super(TAudio, self).__init__()

        # Adapter-based backend
        self.adapter = create_adapter(
            backend=backend, model_id=model_id, load_in_8bit=load_in_8bit)

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
        self.bidirectional_audio = bidirectional_audio

        self.surrogate_loss = surrogate_loss
        self.token_loss = token_loss
        self.surrogate_loss_weight = surrogate_loss_weight
        self.poisson_loss = poisson_loss
        self.task_type = task_type

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
        with self.adapter.bidirectional_audio_context(input_ids) if self.bidirectional_audio else nullcontext():
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
        with self.adapter.bidirectional_audio_context(kwargs['input_ids']) if self.bidirectional_audio else nullcontext():
            return self.adapter.generate(**kwargs)

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
