from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from datasets import load_dataset

model_3b = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-3B")
model_7b = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")

processor_3b = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
processor_7b = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

audioset_human = load_dataset("enyoukai/AudioSet-Strong-Human-Sounds")
libricount = load_dataset("enyoukai/libricount-timings")
librispeech = load_dataset("gilkeyio/librispeech-alignments")