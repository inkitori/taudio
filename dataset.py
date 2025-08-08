import torch
from datasets import Dataset
import random
from typing import Any, Dict, Optional
from nltk.corpus import stopwords
import logging

from utils.utils import clamp, better_round

from datasets import create_adapter, infer_adapter_from_repository
from models.base_model_adapter import BaseModelAdapter
from tasks.types import TaskType

STOPS = set(stopwords.words('english'))


def get_ds(
    model_adapter: BaseModelAdapter,
    repository: str,
    split: str,
    task: TaskType,
    key: Optional[str] = None,
    max_time: Optional[float] = None,
    max_count: Optional[int] = None,
) -> Dataset:
    if task == TaskType.SINGLE_WORD_TIMESTAMP:
        return get_single_word_timestamp_task(
            model_adapter, repository, split, task, key, max_time)
    elif task == TaskType.MULTI_WORD_TIMESTAMP:
        raise ValueError(f"Task type {task} not supported")
    elif task == TaskType.SPEAKER_COUNTING:
        return get_speaker_counting_task(
            model_adapter, repository, split, task, max_count)
    else:
        raise ValueError(f"Unknown task type: {task}")


def get_speaker_counting_task(
    model_adapter: BaseModelAdapter,
    repository: str,
    split: str,
    task: TaskType,
    max_count: Optional[int] = None,
) -> Dataset:
    pass


def get_single_word_timestamp_task(
    model_adapter: BaseModelAdapter,
    repository: str,
    split: str,
    task: TaskType,
    key: Optional[str] = None,
    max_time: Optional[float] = None,
) -> Dataset:
    audio_id = model_adapter.audio_id
    assistant_id = model_adapter.assistant_id

    ds_adapter = create_adapter(infer_adapter_from_repository(repository))

    def preprocess_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        audio = ds_adapter.get_audio(example)
        events = ds_adapter.get_events(example)

        # Get unique words that meet our criteria: not UNK_TOKEN, not stopwords, and occur before max_time seconds
        seen = set()
        candidate_events = []
        for event in events:
            if (
                ds_adapter.event_name(event) not in ds_adapter.unknown_events()
                and ds_adapter.event_name(event) not in STOPS
                and (max_time is None or ds_adapter.get_target_seconds(event, key) < max_time)
                and ds_adapter.event_name(event) not in seen
            ):
                candidate_events.append(event)
            seen.add(ds_adapter.event_name(event))

        if len(candidate_events) > 0:
            event = random.choice(candidate_events)
        else:
            # Fallback to first word if no candidates meet criteria
            event = events[0]
            logging.info(f"No candidates met criteria, using first word: {
                         ds_adapter.event_name(event)}, {ds_adapter.get_target_seconds(event, key)}")

        logging.info(f"Selected Word: {ds_adapter.event_name(event)}, {
                     ds_adapter.get_target_seconds(event, key)}")

        prompt = ds_adapter.build_prompt(
            processor=model_adapter.processor,
            event=event,
            task=task,
            eval_mode=False,
            key=key,
        )

        audio_frames = audio['array']  # 16 khz
        assert audio['sampling_rate'] == 16000

        inputs = model_adapter.processor(
            text=prompt,
            audio=audio_frames,
            return_tensors='pt',
            padding=True,
        )

        input_ids = inputs['input_ids']  # (batch_size, seq_len)
        attention_mask = inputs['attention_mask']  # (batch_size, seq_len)

        # input_features comes in milliseconds. audio_context_len is 30,000 for 30 seconds of audio.
        # (batch_size, embedding_dim, audio_context_len)
        input_features = inputs['input_features']
        # (batch_size, audio_context_len)
        feature_attention_mask = inputs['feature_attention_mask']

        # <AUDIO> tokens in input_ids correspond to audio embeddings (40 ms per embedding)
        labels_size = (input_ids == audio_id).sum().item()
        labels = torch.zeros(labels_size)

        event_idx = clamp(better_round(
            ds_adapter.get_target_seconds(event, key) * model_adapter.seconds_to_embedding), 0, labels_size - 1)
        labels[event_idx] = 1.0

        # mask out everything up to and including the assistant token
        label_ids = input_ids.clone()
        assistant_idx = (input_ids == assistant_id).nonzero(as_tuple=True)[
            1][0]  # first occurence of assistant token
        label_ids[0, :assistant_idx + 1] = -100

        return {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],

            'input_features': input_features[0],
            'feature_attention_mask': feature_attention_mask[0],

            'labels': labels,
            'label_ids': label_ids[0],
        }

    base_ds = ds_adapter.load_streaming_split(split)

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
