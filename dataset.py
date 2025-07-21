import torch
from datasets import load_dataset, Dataset
from transformers import Qwen2_5OmniProcessor
import random
from typing import Any, Dict, Optional
import nltk
from nltk.corpus import stopwords
import numpy as np

from utils import clamp, pad_audio

SECONDS_TO_EMBEDDING = (1000) * (1 / 40) # 40 milliseconds per embedding (from technical report)
# For example, 10 seconds of audio would be 10 * 1000 = 10000 milliseconds, which would be 10000 / 40 = 250 embeddings.

STOPS = set(stopwords.words('english'))
UNK_TOKEN = "<unk>"
ASSISTANT_ID = 77091

def _build_conversation(processor: Qwen2_5OmniProcessor, word: Dict[str, any], key: str,eval: bool = False) -> str:
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
				{"type": "text", "text": f"What is the first occurence of the word '{word['word']}'?"},
				{"type": "audio", "audio": "PLACEHOLDER AUDIO"}, # we will manually fill in the audio
			],
		},
	]

	if not eval:
		word_json = '{"%s": %s}' % (word['word'], word[key])
		conversation.append({
			"role": "assistant",
			"content": [
				{"type": "text", "text": f"{word_json}"},
			],
		})

	text = processor.apply_chat_template(
		conversation,
		tokenize=False,
		add_generation_prompt=eval,
	)

	return text

def get_ds(
		model_id: str, 
		audio_token_id: int, 
		split: str,
		key: str,
		padding: int = 0
) -> Dataset:
	def preprocess_fn(example: Dict[str, Any]) -> Dict[str, Any]:
		audio = example['audio']
		words = example['words']

		candidate_words = [word['word'] for word in words if word['word'] != UNK_TOKEN and word['word'] not in STOPS]
		if len(candidate_words) > 0:
			target_word = random.choice(candidate_words)
		else:
			target_word = words[0]['word']

		print('')
		print(f"Selected Word: {target_word}")

		for word in words:
			if word['word'] == target_word:
				first_occurence = word
				break

		prompt = _build_conversation(processor, first_occurence, key, eval=False)
		audio_frames = audio['array'] # 16 khz

		# Right pad audio_frames by 16 (16 frames per ms) * 40 (40 ms per embedding) * padding zeros
		if padding > 0:
			audio_frames = pad_audio(audio_frames, padding)

		inputs = processor(
			text=prompt,
			audio=audio_frames,
			return_tensors='pt',
			padding=True,
		)

		input_ids = inputs['input_ids'] # (batch_size, seq_len)
		# print(processor.tokenizer.batch_decode(input_ids)[0])
		attention_mask = inputs['attention_mask'] # (batch_size, seq_len)

		# input_features comes in milliseconds. audio_context_len is 30,000 for 30 seconds of audio.
		input_features = inputs['input_features'] # (batch_size, embedding_dim, audio_context_len)
		feature_attention_mask = inputs['feature_attention_mask'] # (batch_size, audio_context_len)


		# <AUDIO> tokens in input_ids correspond to audio embeddings (40 ms per embedding)
		labels_size = (input_ids == audio_token_id).sum().item()
		labels = torch.zeros(labels_size)

		# Left pad labels with zeros if padding was applied
		label_idx_offset = padding if padding > 0 else 0
		idx = clamp(int(first_occurence[key] * SECONDS_TO_EMBEDDING) + label_idx_offset, label_idx_offset, labels_size - 1)

		labels[idx] = 1.0

		# mask out everything up to and including the assistant token
		label_ids = input_ids.clone()
		assistant_idx = (input_ids == ASSISTANT_ID).nonzero(as_tuple=True)[1][0] if (input_ids == ASSISTANT_ID).any() else 0
		label_ids[0, :assistant_idx + 1] = -100

		return {
			'input_ids': input_ids[0],
			'attention_mask': attention_mask[0],

			'input_features': input_features[0],
			'feature_attention_mask': feature_attention_mask[0],

			'labels': labels,
			'label_ids': label_ids[0],
			
			# 'prompt': prompt,
			# 'audio_frames': audio_frames,
			# 'audio_features': audio_features.to('cpu'),
		}

	print(f"Loading processor for {model_id}")
	processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

	base_ds = load_dataset("gilkeyio/librispeech-alignments", split=split, streaming=True)

	ds = base_ds.map(preprocess_fn, remove_columns=base_ds.column_names)

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
		elif key == 'label_ids':
			collated[key] = torch.nn.utils.rnn.pad_sequence(items, batch_first=True, padding_value=-100, padding_side='right')
		else:
			collated[key] = torch.stack(items)

	return collated
