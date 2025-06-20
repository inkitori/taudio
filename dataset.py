import torch
from datasets import load_dataset
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
import random

from utils import clamp

AUDIO_TOKEN_ID = 151646  

def _build_conversation(processor, word):
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
				{"type": "text", "text": f"{word}"},
			],
		},
	]

	text = processor.apply_chat_template(
		conversation,
		tokenize=False,
		add_generation_prompt=True,
	)
	

	return text

def get_ds(model_id, split='train_clean_100', slice=None):
	def preprocess_fn(example):
		audio = example['audio']
		words = example['words']

		word_choices = [w['word'] for w in words]
		random_word = random.choice(word_choices)

		print(f"Selected word: {random_word}")

		for w in words:
			if w['word'] == random_word:
				target_word = w
				break

		prompt = _build_conversation(processor, target_word['word'])
		audio_frames = audio['array']

		inputs = processor(
			text=prompt,
			audio=audio_frames,
			return_tensors='pt',
			padding=True,
		)

		input_ids = inputs['input_ids']
		attention_mask = inputs['attention_mask']
		input_features = inputs['input_features']
		feature_attention_mask = inputs['feature_attention_mask']

		labels_size = (input_ids == AUDIO_TOKEN_ID).sum().item()

		labels = torch.zeros(labels_size)
		end_idx = clamp(int(target_word['end'] * 25), 0, labels_size - 1) # convert to centiseconds and divide by 4
		# TODO: clamp to max size of labels

		labels[end_idx] = 1


		# audio_features = model.get_audio_features(
		# 	input_features=input_features.to(device=model.device),
		# 	feature_attention_mask=feature_attention_mask.to(device=model.device),
		# )

		return {
			# 'prompt': prompt,
			# 'audio_frames': audio_frames,
			'input_ids': input_ids[0],
			'attention_mask': attention_mask[0],
			'input_features': input_features[0],
			'feature_attention_mask': feature_attention_mask[0],
			'labels': labels,
			# 'audio_features': audio_features.to('cpu'),
		}

	processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

	# base_ds = load_dataset("gilkeyio/librispeech-alignments")[split].select(range(slice)) if slice else load_dataset("gilkeyio/librispeech-alignments")[split]
	base_ds = load_dataset("gilkeyio/librispeech-alignments", split=split, streaming=True)

	ds = base_ds.map(preprocess_fn, remove_columns=base_ds.column_names)
	
	ds = ds.with_format('torch')

	return ds
