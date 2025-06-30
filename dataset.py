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
				{"type": "text", "text": f"When is '{word}' said?"},
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
		audio_bos_id: int,
		audio_eos_id: int,
		im_end_id: int,
		unk_token: str,
		split: str = 'train_clean_100'
) -> Dataset:
	def preprocess_fn(example: Dict[str, Any]) -> Dict[str, Any]:
		audio = example['audio']
		words = example['words']
		transcript = example['transcript']

		target_word = random.choice([word['word'] for word in words if word != unk_token])
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

			labels[start_idx] = 1.0
			labels[end_idx] = 1.0

		# --- Start of modification ---
		# The processor returns tensors with a batch dimension of 1, so we squeeze to work with 1D tensors
		ids = input_ids[0]
		mask = attention_mask[0]

		# Find the start and end of the audio token block
		# .nonzero() returns a tensor of indices. We expect exactly one match for BOS and EOS.
		audio_bos_locs = (ids == audio_bos_id).nonzero(as_tuple=True)[0]
		audio_eos_locs = (ids == audio_eos_id).nonzero(as_tuple=True)[0]

		if audio_bos_locs.nelement() != 1 or audio_eos_locs.nelement() != 1:
			raise ValueError(f"Expected one audio_bos and one audio_eos token, but found {audio_bos_locs.nelement()} and {audio_eos_locs.nelement()}.")
		
		audio_bos_idx = audio_bos_locs.item()
		audio_eos_idx = audio_eos_locs.item()

		# Extract the audio block from both input_ids and attention_mask
		audio_block_ids = ids[audio_bos_idx : audio_eos_idx + 1]
		audio_block_mask = mask[audio_bos_idx : audio_eos_idx + 1]

		# Create the remaining tensors by removing the audio block
		remaining_ids = torch.cat([ids[:audio_bos_idx], ids[audio_eos_idx + 1:]])
		remaining_mask = torch.cat([mask[:audio_bos_idx], mask[audio_eos_idx + 1:]])

		# Find the index of the *last* im_end token in the remaining sequence
		im_end_indices = (remaining_ids == im_end_id).nonzero(as_tuple=True)[0]
		if im_end_indices.nelement() == 0:
			raise ValueError(f"Could not find the insertion point token (im_end_id).")
		
		# The insertion point is right before the last im_end token
		insertion_point = im_end_indices[-1].item()

		# Reconstruct the sequences by inserting the audio block before the last im_end_id
		# [ ... text before ... ] + [ audio_block ] + [ ... im_end and after ... ]
		input_ids = torch.cat([
			remaining_ids[:insertion_point],
			audio_block_ids,
			remaining_ids[insertion_point:]
		]).unsqueeze(0) # Add the batch dimension back

		attention_mask = torch.cat([
			remaining_mask[:insertion_point],
			audio_block_mask,
			remaining_mask[insertion_point:]
		]).unsqueeze(0) # Add the batch dimension back
		# --- End of modification ---

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
		else:
			collated[key] = torch.stack(items)

	return collated
