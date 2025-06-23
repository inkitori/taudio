import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen2_5OmniThinkerForConditionalGeneration, BitsAndBytesConfig

class TAudio(nn.Module):
    def __init__(
            self, 
            model_id: str, 
            freeze_text_model: bool, 
            load_in_8bit: bool, 
            audio_layer: int,
            class_weighting: bool,
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

    def get_audio_token_id(self) -> int:
        return self.base_model.config.audio_token_index

    def forward(
        self,
        input_ids: torch.Tensor, # (batch_size, seq_len)
        attention_mask: torch.Tensor, # (batch_size, seq_len)
        input_features: torch.Tensor, # (batch_size, embedding_dim, audio_context_len)
        feature_attention_mask: torch.Tensor, # (batch_size, audio_context_len)
        labels: torch.Tensor # (num_audio_tokens)
    ) -> torch.Tensor:
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states[self.audio_layer]  # (batch_size, seq_len, hidden_dim)

        audio_hidden_states = hidden_states[input_ids == self.get_audio_token_id()]  # (num_audio_tokens, hidden_dim)

        logits = self.linear(audio_hidden_states).squeeze() # (num_audio_tokens)
        labels = labels.to(logits.dtype)

        num_ones = (labels == 1).sum()
        num_zeros = (labels == 0).sum()
        pos_weight = (num_zeros / num_ones) if num_ones > 0 else 1.0

        # print("PREDICTED\t" + str(torch.argmax(pred.squeeze()).item()) + "\t" + str(torch.max(pred.squeeze()).item()) + '\t' + str(pred.sum().item()))
        # print("GROUND TRUTH\t" + str(torch.argmax(labels.squeeze()).item()))
        # print(labels.squeeze() - pred.squeeze())

        # with open('predictions.txt', 'a') as f:
        #     f.write(str(pred.squeeze().tolist()) + '\n')
        #     f.write(str(labels.squeeze().tolist()) + '\n')
        #     f.write('\n')

        if self.class_weighting:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        loss = criterion(logits, labels)

        return loss
