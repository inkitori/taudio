import random
SEED = 80
random.seed(SEED)

import torch
import datasets
from taudio import TAudio
from transformers import Qwen2_5OmniProcessor
import json
import wandb
import argparse
import yaml

split = 'dev_clean'

aux_output = True
token_output = True
error_bound = 0.1

parser = argparse.ArgumentParser(description="Evaluate TAudio model.")
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
parser.add_argument('--config_path', type=str, required=True, help='Path to the model config')
args = parser.parse_args()

with open(args.config_path, 'r') as f:
	model_config = yaml.safe_load(f)

key = model_config.get('key', 'PLACEHOLDER_KEY')

run = wandb.init(
    entity="taudio",
    project="Eval", 
	config={
		"checkpoint_path": args.checkpoint_path,

		"split": split,
		"aux_output": aux_output,
		"error_bound": error_bound,
		"key": key,

		**model_config,
	}
)

audio_layer = model_config['audio_layer']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processor = Qwen2_5OmniProcessor.from_pretrained(model_config['model_id'])
taudio_config = {k: v for k, v in model_config.items() if k != 'key'}
model = TAudio(**taudio_config).to(device)

model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
model.to(device)

base_ds = datasets.load_dataset("gilkeyio/librispeech-alignments", split=split, streaming=True)

aux_correct = 0
token_correct = 0
total = 0

for example in base_ds:
	candidates = {}

	for word in example['words']:
		if word['word'] != "<unk>" and word['word'] not in candidates:
			candidates[word['word']] = word
	
	if not candidates:
		continue

	word = random.choice(list(candidates.values()))

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

	text = processor.apply_chat_template(
		conversation,
		tokenize=False,
		add_generation_prompt=True,
	)

	inputs = processor(
		text=text,
		audio=example['audio']['array'],
		return_tensors='pt',
		padding=True,
	).to(device)

	tokens = model.base_model.generate(
		**inputs,
		eos_token_id = processor.tokenizer.eos_token_id,
	)

	if aux_output:
		outputs = model.base_model(**inputs, output_hidden_states=True)

		hidden_states = outputs.hidden_states[audio_layer]  # (batch_size, seq_len, hidden_dim)
		audio_hidden_states = hidden_states[inputs['input_ids'] == model.get_audio_token_id()]  # (num_audio_tokens, hidden_dim)

		logits = model.linear(audio_hidden_states).squeeze() # (num_audio_tokens)
		_, aux_pred_top_idx = torch.max(logits, dim=0)
	
	gt = word[key]
	total += 1
	
	print(word['word'])
	print("GT: " + str(gt))

	if token_output:
		try:
			generated_tokens = tokens[0][inputs['input_ids'].shape[1]:-1]
			generated_string = processor.tokenizer.decode(generated_tokens)
			token_pred = json.loads(generated_string)[word['word']]
			
			if (abs(token_pred - gt) <= error_bound):
				token_correct += 1
			
			print("PRED: " + str(token_pred))
		except:
			pass

	if aux_output:
		aux_pred = float(aux_pred_top_idx) / 25
		
		if (abs(aux_pred - gt) <= error_bound):
			aux_correct += 1
		
		print("AUX_PRED: " + str(aux_pred))

	run.log({
		"token_accuracy": token_correct / total,
		"auxiliary_accuracy": aux_correct / total
	})


run.finish()
