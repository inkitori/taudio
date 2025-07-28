from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from causal_mask_patch import patch_causal_mask_zero_region
from utils import get_audio_bounds

model_id = "Qwen/Qwen2.5-Omni-3B"
BEGIN_AUDIO_ID = 151647
END_AUDIO_ID = 151648

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
)

processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

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
            {"type": "audio", "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
        ],
    },
]

text = processor.apply_chat_template(
    conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True)
inputs = inputs.to(model.device).to(model.dtype)

patch_causal_mask_zero_region(model.model)

start_audio_index, end_audio_index = get_audio_bounds(
    inputs['input_ids'], BEGIN_AUDIO_ID, END_AUDIO_ID)

model.model.mask_start = start_audio_index
model.model.mask_end = end_audio_index

# outputs = model(
#     input_ids=inputs['input_ids'],
#     attention_mask=inputs['attention_mask'],
#     input_features=inputs['input_features'],
#     feature_attention_mask=inputs['feature_attention_mask'],
# )
tokens = model.base_model.generate(
    **inputs,
    eos_token_id=processor.tokenizer.eos_token_id,
)
