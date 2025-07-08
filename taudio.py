import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen2_5OmniThinkerForConditionalGeneration, BitsAndBytesConfig

from collections import namedtuple

from helpers import poisson_loss

Output = namedtuple('Output', ['loss', 'token_loss', 'surrogate_loss', 'pred', 'gt'])

class TAudio(nn.Module):
    def __init__(
            self, 
            model_id: str, 
            freeze_text_model: bool, 
            load_in_8bit: bool, 
            audio_layer: int,
            class_weighting: bool,
            surrogate_loss: bool,
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

        self.linear = nn.Linear(self.hidden_dim, 1, dtype=self.base_model.dtype)
        self.audio_layer = audio_layer
        self.class_weighting = class_weighting
        self.surrogate_loss = surrogate_loss

    def get_audio_token_id(self) -> int:
        return self.base_model.config.audio_token_index

    def forward(
        self,
        input_ids: torch.Tensor, # (batch_size, seq_len)
        attention_mask: torch.Tensor, # (batch_size, seq_len)
        input_features: torch.Tensor, # (batch_size, embedding_dim, audio_context_len)
        feature_attention_mask: torch.Tensor, # (batch_size, audio_context_len)
        labels: torch.Tensor, # (num_audio_tokens)
        label_ids: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            output_hidden_states=True,
            labels=label_ids,
        )

        hidden_states = outputs.hidden_states[self.audio_layer]  # (batch_size, seq_len, hidden_dim)

        audio_hidden_states = hidden_states[input_ids == self.get_audio_token_id()]  # (num_audio_tokens, hidden_dim)

        logits = self.linear(audio_hidden_states).squeeze() # (num_audio_tokens)
        labels = labels.to(logits.dtype)

		# TODO: fix this code for batched inputs
        pred_top_val, pred_top_idx = torch.max(logits, dim=0)
        gt_top_val, gt_top_idx = torch.max(labels, dim=0)

        if self.class_weighting:
            num_ones = (labels == 1).sum()
            num_zeros = (labels == 0).sum()
            pos_weight = (num_zeros / num_ones) if num_ones > 0 else 1.0

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        if self.surrogate_loss:
            surrogate_loss = criterion(logits, labels)
        else:
            surrogate_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        token_loss = outputs.loss

        loss = surrogate_loss + token_loss

        output = Output(loss=loss, surrogate_loss=surrogate_loss, token_loss=token_loss, pred=(pred_top_val, pred_top_idx), gt=(gt_top_val, gt_top_idx))
        
        # logits = logits.unsqueeze(0)
        # labels = labels.unsqueeze(0)
        # loss = poisson_loss(logits, labels, torch.ones_like(logits))

        return output
