import torch
import datasets
from taudio import TAudio
from transformers import Qwen2_5OmniProcessor
import json
import wandb
import argparse
import random

SEED = 80

random.seed(SEED)

parser = argparse.ArgumentParser(description="Evaluate TAudio model.")
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
args = parser.parse_args()

checkpoint_path = args.checkpoint_path
split = 'test_clean'
aux_output = False
error_bound = 0.1

run = wandb.init(
    entity="taudio",
    project="Eval", 
	config={
		"checkpoint_path": checkpoint_path,
		"split": split,
		"aux_output": aux_output,
		"error_bound": error_bound
	}
)

model_id = "Qwen/Qwen2.5-Omni-3B"
freeze_text_model = False
load_in_8bit = False
audio_layer = -1 # which layer of the text model to project down to score
class_weighting = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
model = TAudio(
    model_id=model_id, 
    freeze_text_model=freeze_text_model, 
    load_in_8bit=load_in_8bit,
    audio_layer=audio_layer,
    class_weighting=class_weighting,
    surrogate_loss=True
).to(device)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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
	
	try:
		generated_tokens = tokens[0][inputs['input_ids'].shape[1]:-1]
		generated_string = processor.tokenizer.decode(generated_tokens)

		token_pred = json.loads(generated_string)[word['word']]

		if aux_output:
			aux_pred = float(aux_pred_top_idx) / 25

		gt = word['end']

		if (abs(token_pred - gt) <= error_bound):
			token_correct += 1
		
		if aux_output:
			if (abs(aux_pred - gt) <= error_bound):
				aux_correct += 1

		total += 1

		print(word['word'])
		print("GT: " + str(gt))
		print("PRED: " + str(token_pred))

		if aux_output:
			print("AUX_PRED: " + str(aux_pred))

	except:
		total += 1

	run.log({
		"token_accuracy": token_correct / total,
		"auxiliary_accuracy": aux_correct / total
	})


run.finish()