import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen2_5OmniThinkerForConditionalGeneration, BitsAndBytesConfig
from causal_mask_patch import patch_causal_mask_zero_region

from collections import namedtuple
from typing import Optional

from utils import get_audio_bounds
from helpers import poisson_loss
import logging

Output = namedtuple(
    'Output', ['loss', 'token_loss', 'surrogate_loss', 'pred', 'gt'])

BEGIN_AUDIO_ID = 151647
END_AUDIO_ID = 151648


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
            linear_bias: Optional[float] = None
    ) -> None:
        super(TAudio, self).__init__()

        self.base_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            load_in_8bit=load_in_8bit,
        )
        self.base_audio_encoder = self.base_model.audio_tower
        self.base_text_model = self.base_model.model

        patch_causal_mask_zero_region(self.base_text_model)

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
        label_ids: torch.Tensor
    ) -> torch.Tensor:
        if self.bidirectional_audio:
            start_audio_index, end_audio_index = get_audio_bounds(
                input_ids, BEGIN_AUDIO_ID, END_AUDIO_ID)

            self.base_text_model.mask_start = start_audio_index
            self.base_text_model.mask_end = end_audio_index

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
            if self.poisson_loss:
                surrogate_loss = poisson_loss(logits.unsqueeze(0), labels.unsqueeze(0), torch.ones_like(logits.unsqueeze(0)))
            else:
                if self.class_weighting:
                    num_ones = (labels == 1).sum()
                    num_zeros = (labels == 0).sum()
                    pos_weight = (num_zeros / num_ones) if num_ones > 0 else 1.0

                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                else:
                    criterion = nn.BCEWithLogitsLoss()
                
                surrogate_loss = criterion(logits, labels)
        else:
            surrogate_loss = torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype)

        if self.token_loss:
            token_loss = outputs.loss
        else:
            token_loss = torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype)

        loss = self.surrogate_loss_weight * surrogate_loss + token_loss

        output = Output(loss=loss, surrogate_loss=surrogate_loss, token_loss=token_loss, pred=(
            pred_top_val, pred_top_idx), gt=(gt_top_val, gt_top_idx))

        return output
