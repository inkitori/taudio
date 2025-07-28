from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from causal_mask_patch import patch_causal_mask_zero_region, unpatch_causal_mask

model_id = "Qwen/Qwen2.5-Omni-3B"

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
)

processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

# Apply the causal mask patch
# This will zero out the rectangular region from position 5 to 15 in the causal mask
patch_causal_mask_zero_region(model.model, start=5, end=15)

# You can also set the values later or change them:
# model.model.mask_start = 10
# model.model.mask_end = 20

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

# Now when you run the model, the causal mask will have the specified region zeroed out
outputs = model(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    input_features=inputs['input_features'],
    feature_attention_mask=inputs['feature_attention_mask'],
)

print(outputs)

# To disable the zeroing temporarily:
# model.model.mask_start = None
# model.model.mask_end = None

# To change the region:
# model.model.mask_start = 8
# model.model.mask_end = 12

# To completely remove the patch and restore original behavior:
# unpatch_causal_mask(model.model)
