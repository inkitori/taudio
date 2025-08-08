from typing import Any, Dict, Iterable, List, Optional
from datasets import load_dataset

from tasks.types import TaskType

from .base_dataset_adapter import BaseDatasetAdapter


class LibriSpeechAdapter(BaseDatasetAdapter):
    def load_streaming_split(self, split: str):
        return load_dataset(self.repository, split=split, streaming=True)

    def get_audio(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return example["audio"]

    def get_events(self, example: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        # The source data uses key 'words', each item has 'word' and timing fields like 'start'/'end'
        return example["words"]

    def event_name(self, event: Dict[str, Any]) -> str:
        # There are <unk> tokens which the generic pipeline can filter if desired
        return event.get("word", "")

    def get_target_seconds(self, event: Dict[str, Any], key: str) -> float:
        # key could be 'start' or 'end'
        return float(event[key])

    def build_prompt(self, model_processor: Any, event: Dict[str, Any], task: TaskType, eval_mode: bool, key: Optional[str]) -> str:
        # Reuse semantics of the existing conversation format used in the current code
        name = event.get("word", "")
        if task == TaskType.SINGLE_WORD_TIMESTAMP:
            prompt = f"What is the first occurence of the word '{name}'?"
        elif task == TaskType.MULTI_WORD_TIMESTAMP:
            prompt = f"When do occurrences of the word '{name}' happen?"
        elif task == TaskType.SPEAKER_COUNTING:
            prompt = "How many distinct speakers are there?"
        else:
            prompt = f"Task: {task}. Event: {name}"

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "audio": "PLACEHOLDER AUDIO"},
                ],
            },
        ]

        if not eval_mode and key is not None and task.endswith("timestamp"):
            # Supervision as JSON like {"<word>": <seconds>}
            word_json = '{"%s": %s}' % (name, event[key])
            conversation.append(
                {"role": "assistant", "content": [
                    {"type": "text", "text": f"{word_json}"}]}
            )

        return model_processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=eval_mode
        )

    def unknown_events(self) -> List[str]:
        return ["<unk>"]
