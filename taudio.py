import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen2_5OmniThinkerForConditionalGeneration

class TAudio(nn.Module):
	def __init__(self, model_id, freeze_model=False):
		super(TAudio, self).__init__()

		self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
		self.hidden_dim = self.model.config.audio_config.output_dim

		if freeze_model:
			for param in self.model.parameters():
				param.requires_grad = False

		self.linear = nn.Linear(self.hidden_dim, 1, dtype=self.model.dtype)

	def forward(self, input_ids, attention_mask, input_features, feature_attention_mask, labels):
		outputs = self.model(
			input_ids=input_ids, 
			attention_mask=attention_mask,
			input_features=input_features,
			feature_attention_mask=feature_attention_mask,
			output_hidden_states=True
		)

		last_layer_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)

		audio_hidden_states = last_layer_hidden_state[input_ids == self.model.config.audio_token_index]  # (num_audio_tokens, hidden_dim)

		# do some stuff to segment only the audio embeddings (i think we can just use the features attention mask)

		pred = F.sigmoid(self.linear(audio_hidden_states))

		labels = labels.to(pred.dtype)

		# print(pred)

		print(' ')
		print("PREDICTED\t" + str(torch.argmax(pred.squeeze()).item()) + "\t" + str(torch.max(pred.squeeze()).item()))
		print("GROUND TRUTH\t" + str(torch.argmax(labels.squeeze()).item()))

		loss = F.binary_cross_entropy(pred.squeeze(), labels.squeeze(), reduction='none')
		loss[torch.argmax(labels.squeeze()).item()] *= 10
		return loss.mean()
