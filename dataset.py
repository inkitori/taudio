# this file only works for Qwen2.5 Omni

import torch
from datasets import load_dataset, Dataset
from transformers import Qwen2_5OmniProcessor
import random
from typing import Any, Dict, Optional
import nltk
from nltk.corpus import stopwords
import numpy as np
import logging

from utils import clamp
from qwen2_5_omni_constants import ASSISTANT_ID, SECONDS_TO_EMBEDDING

STOPS = set(stopwords.words('english'))
UNK_TOKEN = "<unk>" # this only applies to librispeech

def build_conversation(processor: Qwen2_5OmniProcessor, repository: str, word: Dict[str, any], key: str, eval: bool) -> str:
    if repository == "gilkeyio/librispeech-alignments":
        prompt = f"What is the first occurence of the word '{word['word']}'?"
    elif repository == "enyoukai/audiotime-timestamps":
        prompt = f"What is the first occurence of '{word['word']}'?" # we call these "words" but they're just general events
    else:
        raise ValueError(f"Invalid repository: {repository}")

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
                {"type": "text", "text": prompt},
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
    repository: str = "gilkeyio/librispeech-alignments",
    max_time: Optional[float] = None,
) -> Dataset:
    def preprocess_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        audio = example['audio']
        words = example['words']

        # Get unique words that meet our criteria: not UNK_TOKEN, not stopwords, and occur before max_time seconds
        seen = set()
        candidate_words = []
        for word in words:
            if word['word'] != UNK_TOKEN and word['word'] not in STOPS and (max_time is None or word[key] < max_time) and word['word'] not in seen:
                candidate_words.append(word)

            seen.add(word['word'])

        if len(candidate_words) > 0:
            word = random.choice(candidate_words)
        else:
            # Fallback to first word if no candidates meet criteria
            word = words[0]
            logging.info(f"No candidates met criteria, using first word: {word['word']}, {word[key]}")

        logging.info(f"Selected Word: {word['word']}, {word[key]}")

        prompt = build_conversation(
            processor=processor,
            repository=repository,
            word=word,
            key=key,
            eval=False,
        )

        audio_frames = audio['array']  # 16 khz
        assert audio['sampling_rate'] == 16000

        inputs = processor(
            text=prompt,
            audio=audio_frames,
            return_tensors='pt',
            padding=True,
        )

        input_ids = inputs['input_ids']  # (batch_size, seq_len)
        attention_mask = inputs['attention_mask']  # (batch_size, seq_len)

        # input_features comes in milliseconds. audio_context_len is 30,000 for 30 seconds of audio.
        input_features = inputs['input_features'] # (batch_size, audio_context_len)
        feature_attention_mask = inputs['feature_attention_mask'] # (batch_size, audio_context_len)

        # <AUDIO> tokens in input_ids correspond to audio embeddings (40 ms per embedding)
        labels_size = (input_ids == audio_token_id).sum().item()
        labels = torch.zeros(labels_size)

        idx = clamp(int(word[key] * SECONDS_TO_EMBEDDING), 0, labels_size - 1)

        labels[idx] = 1.0

        # mask out everything up to and including the assistant token
        label_ids = input_ids.clone()
        assistant_idx = (input_ids == ASSISTANT_ID).nonzero(as_tuple=True)[
            1][0] if (input_ids == ASSISTANT_ID).any() else 0
        label_ids[0, :assistant_idx + 1] = -100

        return {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],

            'input_features': input_features[0],
            'feature_attention_mask': feature_attention_mask[0],

            'labels': labels,
            'label_ids': label_ids[0],
        }

    logging.info(f"Loading processor for {model_id}")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    base_ds = load_dataset(repository, split=split, streaming=True)

    ds = base_ds.map(preprocess_fn, remove_columns=base_ds.column_names)

    return ds


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    batch_keys = batch[0].keys()
    collated = {}

    for key in batch_keys:
        items = [item[key] for item in batch]
        if key == 'input_ids' or key == 'attention_mask':
            collated[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=0, padding_side='left')
        elif key == 'labels':
            collated[key] = torch.cat(items, dim=0)
        elif key == 'label_ids':
            collated[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=-100, padding_side='right')
        else:
            collated[key] = torch.stack(items)

    return collated
