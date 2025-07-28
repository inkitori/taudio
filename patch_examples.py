from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# Import different patching approaches
from causal_mask_patch import patch_causal_mask_zero_region, unpatch_causal_mask
from advanced_causal_mask_patch import (
    patch_causal_mask_advanced,
    create_rectangular_mask_fn,
    create_pattern_mask_fn,
    patch_rectangular_regions,
    patch_diagonal_mask,
    patch_band_mask
)


def load_model_and_data():
    """Helper function to load model and prepare data"""
    model_id = "Qwen/Qwen2.5-Omni-3B"

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human."}
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
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=True)
    inputs = processor(text=text, audio=audios,
                       return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)

    return model, inputs


def example_1_simple_rectangular():
    """Example 1: Simple rectangular region zeroing"""
    print("=== Example 1: Simple rectangular region ===")

    model, inputs = load_model_and_data()

    # Patch to zero out region from token 10 to 20
    patch_causal_mask_zero_region(model.model, start=10, end=20)

    # Run inference
    outputs = model(**inputs)
    print("Inference completed with rectangular mask")

    # Clean up
    unpatch_causal_mask(model.model)


def example_2_dynamic_control():
    """Example 2: Dynamic control of mask regions"""
    print("=== Example 2: Dynamic control ===")

    model, inputs = load_model_and_data()

    # Apply patch without initial values
    patch_causal_mask_zero_region(model.model)

    # Set mask region dynamically
    model.model.mask_start = 5
    model.model.mask_end = 15
    outputs1 = model(**inputs)
    print("First inference with mask [5, 15]")

    # Change mask region
    model.model.mask_start = 20
    model.model.mask_end = 30
    outputs2 = model(**inputs)
    print("Second inference with mask [20, 30]")

    # Disable masking
    model.model.mask_start = None
    model.model.mask_end = None
    outputs3 = model(**inputs)
    print("Third inference with no masking")


def example_3_multiple_regions():
    """Example 3: Multiple rectangular regions"""
    print("=== Example 3: Multiple regions ===")

    model, inputs = load_model_and_data()

    # Define multiple regions to zero out
    regions = [(5, 10), (15, 20), (25, 30)]
    patch_rectangular_regions(model.model, regions)

    outputs = model(**inputs)
    print(f"Inference completed with multiple regions: {regions}")


def example_4_custom_mask_function():
    """Example 4: Custom mask function"""
    print("=== Example 4: Custom mask function ===")

    model, inputs = load_model_and_data()

    # Define a custom mask function that creates a checkerboard pattern
    def checkerboard_mask(causal_mask, batch_idx, seq_len):
        if causal_mask is not None and causal_mask.dim() >= 4:
            for i in range(0, causal_mask.shape[2], 4):
                for j in range(0, causal_mask.shape[3], 4):
                    if (i // 4 + j // 4) % 2 == 0:  # Checkerboard pattern
                        end_i = min(i + 2, causal_mask.shape[2])
                        end_j = min(j + 2, causal_mask.shape[3])
                        causal_mask[batch_idx, 0, i:end_i, j:end_j] = 0
        return causal_mask

    patch_causal_mask_advanced(model.model, mask_fn=checkerboard_mask)

    outputs = model(**inputs)
    print("Inference completed with checkerboard pattern")


def example_5_predefined_patterns():
    """Example 5: Predefined patterns"""
    print("=== Example 5: Predefined patterns ===")

    model, inputs = load_model_and_data()

    # Diagonal mask
    patch_diagonal_mask(model.model, width=3)
    outputs1 = model(**inputs)
    print("Inference with diagonal mask")

    # Reset and try band mask
    unpatch_causal_mask(model.model)
    patch_band_mask(model.model, band_size=5, skip_size=5)
    outputs2 = model(**inputs)
    print("Inference with band mask")


def example_6_conditional_masking():
    """Example 6: Conditional masking based on sequence properties"""
    print("=== Example 6: Conditional masking ===")

    model, inputs = load_model_and_data()

    def conditional_mask(causal_mask, batch_idx, seq_len):
        if causal_mask is not None and causal_mask.dim() >= 4:
            # Only apply masking if sequence is long enough
            if seq_len > 50:
                # Mask middle third of the sequence
                start = seq_len // 3
                end = 2 * seq_len // 3
                causal_mask[batch_idx, 0, start:end, start:end] = 0
                print(f"Applied conditional mask for batch {
                      batch_idx}, seq_len {seq_len}: [{start}:{end}]")
        return causal_mask

    patch_causal_mask_advanced(model.model, mask_fn=conditional_mask)

    outputs = model(**inputs)
    print("Inference completed with conditional masking")


if __name__ == "__main__":
    # Run all examples
    example_1_simple_rectangular()
    print()

    example_2_dynamic_control()
    print()

    example_3_multiple_regions()
    print()

    example_4_custom_mask_function()
    print()

    example_5_predefined_patterns()
    print()

    example_6_conditional_masking()
    print()

    print("All examples completed!")
