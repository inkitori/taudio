from io import BytesIO
from urllib.request import urlopen
import librosa
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-3B")
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")

conversations = [
        {"role": "user", "content": [
            {"type": "text", "text": "How are you doing right now?"},
        ]},
]

text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, return_tensors="pt", padding=True)

# Generate
generation = thinker.generate(**inputs, max_new_tokens=32)
generate_ids = generation[:, inputs.input_ids.size(1):]

response = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

print(generation)
print(generate_ids)
print(response)