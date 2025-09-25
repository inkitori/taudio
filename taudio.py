from collections import namedtuple
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from models import create_adapter
from tasks.base_task import BaseTask

Output = namedtuple(
    'Output', ['loss', 'token_loss', 'surrogate_loss', 'auxiliary_deviation', 'auxiliary_prediction'])


class TAudio(nn.Module):
    def __init__(
        self,
        model_id: str,
        freeze_text_model: bool,
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
    ) -> None:
        super(TAudio, self).__init__()

        # Adapter-based backend
        self.model_adapter = create_adapter(model_id=model_id, bidirectional_audio=bidirectional_audio, dtype=dtype)

        self.hidden_dim = self.model_adapter.hidden_dim

        if freeze_text_model:
            for param in self.model_adapter.text_model.parameters():
                param.requires_grad = False

        self.linear = nn.Linear(self.hidden_dim, 1, dtype=self.model_adapter.dtype)

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

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch_size, seq_len)
        attention_mask: torch.Tensor,  # (batch_size, seq_len)
        # (batch_size, embedding_dim, audio_context_len)
        input_features: torch.Tensor,
        # (batch_size, audio_context_len)
        feature_attention_mask: torch.Tensor,
        labels: torch.Tensor = None,  # (batch_size, seq_len)
        audio_labels: torch.Tensor = None,  # (batch_size, num_audio_tokens) or (num_audio_tokens)
        inference: bool = False,
    ) -> torch.Tensor:
        if not inference and (labels is None or audio_labels is None):
            raise ValueError("Labels and audio labels must be provided if not in inference mode")

        
        batch_size = input_ids.size(0)

        if self.token_loss:
            outputs = self.model_adapter(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                output_hidden_states=True,
                labels=labels,
            )
        else:
            outputs = self.model_adapter(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                output_hidden_states=True,
            )

        # (batch_size, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states[self.audio_layer]

        audio_logits = []

        for example in range(batch_size):
            audio_hidden_states = hidden_states[example][input_ids[example] == self.model_adapter.audio_id] # (num_audio_tokens, hidden_dim)
            example_audio_logits = self.linear(audio_hidden_states).squeeze() # (num_audio_tokens,)
            audio_logits.append(example_audio_logits)

        audio_logits = torch.nn.utils.rnn.pad_sequence(audio_logits, batch_first=True, padding_value=0, padding_side='right') # (batch_size, num_audio_tokens)
        
        if inference:
            audio_labels = torch.zeros_like(audio_logits)

        audio_labels_frame_mask = torch.where(audio_labels == -100, 0, 1) # (batch_size, num_audio_tokens)

        if self.surrogate_loss:
            surrogate_loss, auxiliary_deviation, auxiliary_prediction = self.task.calculate_loss(
                audio_logits=audio_logits, 
                audio_labels=audio_labels, 
                audio_labels_frame_mask=audio_labels_frame_mask, 
                model_adapter=self.model_adapter, 
                use_poisson_loss=self.poisson_loss, 
                class_weighting=self.class_weighting
            )
        else:
            surrogate_loss = torch.tensor(
                0.0, device=audio_logits.device, dtype=audio_logits.dtype)
            auxiliary_deviation = torch.tensor(
                0.0, device=audio_logits.device, dtype=audio_logits.dtype)
            auxiliary_prediction = torch.tensor(
                0.0, device=audio_logits.device, dtype=audio_logits.dtype)

        if self.token_loss:
            token_loss = outputs.loss
        else:
            token_loss = torch.tensor(
                0.0, device=audio_logits.device, dtype=audio_logits.dtype)

        loss = token_loss + self.surrogate_loss_weight * surrogate_loss

        output = Output(loss=loss, surrogate_loss=surrogate_loss, token_loss=token_loss, auxiliary_deviation=auxiliary_deviation, auxiliary_prediction=auxiliary_prediction)

        return output

    def generate(self, **kwargs):
        with torch.no_grad():
            tokens = self.model_adapter.generate(**kwargs, eos_token_id=self.model_adapter.processor.tokenizer.eos_token_id)

            generated_tokens = tokens[0][kwargs["input_ids"].shape[1]:-1]
            generated_string = self.model_adapter.processor.tokenizer.decode(generated_tokens)

        return generated_string