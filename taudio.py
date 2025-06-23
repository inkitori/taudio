import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen2_5OmniThinkerForConditionalGeneration, BitsAndBytesConfig

class TAudio(nn.Module):
    def __init__(self, model_id: str, freeze_text_model: bool, load_in_8bit: bool) -> None:
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

    def get_audio_token_id(self) -> int:
        return self.base_model.config.audio_token_index

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            output_hidden_states=True
        )

        last_layer_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)

        audio_hidden_states = last_layer_hidden_state[input_ids == self.get_audio_token_id()]  # (num_audio_tokens, hidden_dim)

        logits = self.linear(audio_hidden_states)
        pred = torch.sigmoid(logits)

        labels = labels.to(logits.dtype)

        num_ones = (labels == 1).sum().item()
        num_zeros = (labels == 0).sum().item()
        pos_weight = (num_zeros / num_ones) if num_ones > 0 else 1.0

        # print("PREDICTED\t" + str(torch.argmax(pred.squeeze()).item()) + "\t" + str(torch.max(pred.squeeze()).item()) + '\t' + str(pred.sum().item()))
        # print("GROUND TRUTH\t" + str(torch.argmax(labels.squeeze()).item()))
        # print(labels.squeeze() - pred.squeeze())

        # with open('predictions.txt', 'a') as f:
        #     f.write(str(pred.squeeze().tolist()) + '\n')
        #     f.write(str(labels.squeeze().tolist()) + '\n')
        #     f.write('\n')

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype))
        loss = criterion(logits.squeeze(), labels.squeeze())

        return loss
