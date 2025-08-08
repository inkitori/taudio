from typing import Any, Dict, Iterable, Optional, List
from datasets import load_dataset

from tasks.types import TaskType

from .base_dataset_adapter import BaseDatasetAdapter


class AudioTimeAdapter(BaseDatasetAdapter):
    def load_streaming_split(self, split: str):
        return load_dataset(self.repository, split=split, streaming=True)

    def get_audio(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return example["audio"]

    def get_events(self, example: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        # In AudioTime, we call these "words" but they are general events with a 'word' label and timing
        return example["words"]

    def event_name(self, event: Dict[str, Any]) -> str:
        return event.get("word", "")

    def get_target_seconds(self, event: Dict[str, Any], key: str) -> float:
        return float(event[key])

    def build_prompt(self, model_processor: Any, event: Dict[str, Any], task: TaskType, eval_mode: bool, key: Optional[str]) -> str:
        name = event.get("word", "")
        if task == TaskType.SINGLE_WORD_TIMESTAMP:
            prompt = f"What is the first occurence of '{name}'?"
        elif task == TaskType.MULTI_WORD_TIMESTAMP:
            prompt = f"When do occurrences of '{name}' happen?"
        elif task == TaskType.EVENT_COUNTING:
            prompt = f"How many times does '{name}' occur?"
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
            word_json = '{"%s": %s}' % (name, event[key])
            conversation.append(
                {"role": "assistant", "content": [
                    {"type": "text", "text": f"{word_json}"}]}
            )

        return model_processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=eval_mode
        )

    def unknown_events(self) -> List[str]:
        return []
