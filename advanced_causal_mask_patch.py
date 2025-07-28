import types
import torch
from typing import Union, List, Tuple, Callable


def patch_causal_mask_advanced(model, mask_fn=None, **kwargs):
    """
    Advanced patching of causal mask with flexible masking functions.

    Args:
        model: The model instance to patch
        mask_fn: A function that takes (causal_mask, batch_idx, seq_len) and returns modified mask
        **kwargs: Additional parameters stored on model instance

    Usage:
        # Simple rectangular region
        def zero_region(causal_mask, batch_idx, seq_len):
            if causal_mask is not None and causal_mask.dim() >= 4:
                causal_mask[batch_idx, 0, 10:20, 10:20] = 0
            return causal_mask

        patch_causal_mask_advanced(model.model, mask_fn=zero_region)

        # Multiple regions
        def zero_multiple_regions(causal_mask, batch_idx, seq_len):
            if causal_mask is not None and causal_mask.dim() >= 4:
                causal_mask[batch_idx, 0, 5:10, 5:10] = 0
                causal_mask[batch_idx, 0, 15:25, 15:25] = 0
            return causal_mask

        # Diagonal masking
        def zero_diagonal(causal_mask, batch_idx, seq_len):
            if causal_mask is not None and causal_mask.dim() >= 4:
                for i in range(min(seq_len, causal_mask.shape[2])):
                    causal_mask[batch_idx, 0, i, i] = 0
            return causal_mask
    """

    # Store the original method
    original_method = model._update_causal_mask

    # Set the mask function and any additional parameters
    model.mask_fn = mask_fn
    for key, value in kwargs.items():
        setattr(model, f'mask_{key}', value)

    def patched_update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions=False):
        # Call the original method
        causal_mask = original_method(
            attention_mask, input_tensor, cache_position, past_key_values, output_attentions)

        # Apply custom mask function if provided
        if causal_mask is not None and hasattr(self, 'mask_fn') and self.mask_fn is not None:
            batch_size = causal_mask.shape[0] if causal_mask.dim() >= 4 else 1
            seq_len = causal_mask.shape[2] if causal_mask.dim() >= 4 else 0

            for batch_idx in range(batch_size):
                causal_mask = self.mask_fn(causal_mask, batch_idx, seq_len)

        return causal_mask

    # Replace the method on the instance
    model._update_causal_mask = types.MethodType(
        patched_update_causal_mask, model)

    print(f"Applied advanced causal mask patch to {type(model).__name__}")


def create_rectangular_mask_fn(regions: List[Tuple[int, int]]):
    """
    Create a mask function that zeros out rectangular regions.

    Args:
        regions: List of (start, end) tuples defining regions to zero out

    Returns:
        A mask function that can be used with patch_causal_mask_advanced
    """
    def mask_fn(causal_mask, batch_idx, seq_len):
        if causal_mask is not None and causal_mask.dim() >= 4:
            for start, end in regions:
                if 0 <= start <= end < causal_mask.shape[2]:
                    causal_mask[batch_idx, 0, start:end+1, start:end+1] = 0
        return causal_mask

    return mask_fn


def create_pattern_mask_fn(pattern: str, **params):
    """
    Create predefined mask patterns.

    Args:
        pattern: One of 'diagonal', 'bands', 'checkerboard', 'blocks'
        **params: Pattern-specific parameters

    Returns:
        A mask function for the specified pattern
    """
    if pattern == 'diagonal':
        width = params.get('width', 1)

        def mask_fn(causal_mask, batch_idx, seq_len):
            if causal_mask is not None and causal_mask.dim() >= 4:
                for i in range(causal_mask.shape[2]):
                    for j in range(max(0, i-width//2), min(causal_mask.shape[3], i+width//2+1)):
                        causal_mask[batch_idx, 0, i, j] = 0
            return causal_mask

    elif pattern == 'bands':
        band_size = params.get('band_size', 5)
        skip_size = params.get('skip_size', 5)

        def mask_fn(causal_mask, batch_idx, seq_len):
            if causal_mask is not None and causal_mask.dim() >= 4:
                for start in range(0, causal_mask.shape[2], band_size + skip_size):
                    end = min(start + band_size, causal_mask.shape[2])
                    causal_mask[batch_idx, 0, start:end, start:end] = 0
            return causal_mask

    elif pattern == 'blocks':
        block_size = params.get('block_size', 8)
        stride = params.get('stride', None) or block_size

        def mask_fn(causal_mask, batch_idx, seq_len):
            if causal_mask is not None and causal_mask.dim() >= 4:
                for i in range(0, causal_mask.shape[2], stride):
                    for j in range(0, causal_mask.shape[3], stride):
                        end_i = min(i + block_size, causal_mask.shape[2])
                        end_j = min(j + block_size, causal_mask.shape[3])
                        causal_mask[batch_idx, 0, i:end_i, j:end_j] = 0
            return causal_mask

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return mask_fn


# Convenience functions for common use cases
def patch_rectangular_regions(model, regions: List[Tuple[int, int]]):
    """Patch model to zero out rectangular regions."""
    mask_fn = create_rectangular_mask_fn(regions)
    patch_causal_mask_advanced(model, mask_fn=mask_fn)


def patch_diagonal_mask(model, width: int = 1):
    """Patch model to zero out diagonal regions."""
    mask_fn = create_pattern_mask_fn('diagonal', width=width)
    patch_causal_mask_advanced(model, mask_fn=mask_fn)


def patch_band_mask(model, band_size: int = 5, skip_size: int = 5):
    """Patch model to zero out alternating bands."""
    mask_fn = create_pattern_mask_fn(
        'bands', band_size=band_size, skip_size=skip_size)
    patch_causal_mask_advanced(model, mask_fn=mask_fn)
