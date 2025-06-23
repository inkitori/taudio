import torch
from datasets import load_dataset, Dataset
from transformers import Qwen2_5OmniProcessor
import random
from typing import Any, Dict, Optional

from utils import clamp

SECONDS_TO_EMBEDDING = (1000) * (1 / 40) # 40 milliseconds per embedding (from technical report)
# For example, 10 seconds of audio would be 10 * 1000 = 10000 milliseconds, which would be 10000 / 40 = 250 embeddings.

def _build_conversation(processor: Qwen2_5OmniProcessor, transcript: str, word: str) -> str:
	conversation = [
		{
			"role": "system",
			"content": [
				{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
			],
		},
		{
			"role": "user",
			"content": [
				{"type": "audio", "audio": "PLACEHOLDER AUDIO"}, # we will manually fill in the audio
				{"type": "text", "text": f"Transcript: {transcript}\n Based on the above transcript, when is '{word}' said?"},
			],
		},
	]

	text = processor.apply_chat_template(
		conversation,
		tokenize=False,
		add_generation_prompt=True,
	)

	return text

def get_ds(
		model_id: str, 
		audio_token_id: int, 
		split: str = 'train_clean_100'
) -> Dataset:
	def preprocess_fn(example: Dict[str, Any]) -> Dict[str, Any]:
		audio = example['audio']
		words = example['words']
		transcript = example['transcript']

		target_word = random.choice([word['word'] for word in words])
		occurences = [word for word in words if word['word'] == target_word]

		prompt = _build_conversation(processor, transcript, target_word)
		audio_frames = audio['array'] # 16 khz

		inputs = processor(
			text=prompt,
			audio=audio_frames,
			return_tensors='pt',
			padding=True,
		)

		input_ids = inputs['input_ids'] # (batch_size, seq_len)
		attention_mask = inputs['attention_mask'] # (batch_size, seq_len)

		# input_features comes in milliseconds. audio_context_len is 30,000 for 30 seconds of audio.
		input_features = inputs['input_features'] # (batch_size, embedding_dim, audio_context_len)
		feature_attention_mask = inputs['feature_attention_mask'] # (batch_size, audio_context_len)

		# <AUDIO> tokens in input_ids correspond to audio embeddings (40 ms per embedding)
		labels_size = (input_ids == audio_token_id).sum().item()
		labels = torch.zeros(labels_size)

		for w in occurences:
			start_idx = clamp(int(w['start'] * SECONDS_TO_EMBEDDING), 0, labels_size - 1)
			end_idx = clamp(int(w['end'] * SECONDS_TO_EMBEDDING), 0, labels_size - 1)

			labels[start_idx:end_idx + 1] = 1.0

		return {
			'input_ids': input_ids[0],
			'attention_mask': attention_mask[0],

			'input_features': input_features[0],
			'feature_attention_mask': feature_attention_mask[0],

			'labels': labels,
			
			# 'prompt': prompt,
			# 'audio_frames': audio_frames,
			# 'audio_features': audio_features.to('cpu'),
		}

	processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

	base_ds = load_dataset("gilkeyio/librispeech-alignments", split=split, streaming=True)

	ds = base_ds.map(preprocess_fn, remove_columns=base_ds.column_names).with_format('torch')

	return ds

def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
	batch_keys = batch[0].keys()
	collated = {}

	for key in batch_keys:
		items = [item[key] for item in batch]
		if key == 'input_ids' or key == 'attention_mask':
			collated[key] = torch.nn.utils.rnn.pad_sequence(items, batch_first=True, padding_value=0, padding_side='left')
		elif key == 'labels':
			collated[key] = torch.cat(items, dim=0)
		else:
			collated[key] = torch.stack(items)

	return collated
