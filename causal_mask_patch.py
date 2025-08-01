import types
import torch
import logging


def patch_causal_mask_zero_region(model, start=None, end=None):
    """
    Patch a Qwen2_5OmniThinkerTextModel instance to zero out a rectangular region in the causal mask.

    Args:
        model: The model instance to patch (should be Qwen2_5OmniThinkerTextModel)
        start: Start index for the region to zero out (can be set later via model.mask_start)
        end: End index for the region to zero out (can be set later via model.mask_end)

    Usage:
        # Apply the patch
        patch_causal_mask_zero_region(model, start=10, end=20)

        # Or set later
        patch_causal_mask_zero_region(model)
        model.mask_start = 10
        model.mask_end = 20

        # To disable the zeroing, set either to None
        model.mask_start = None
        model.mask_end = None
    """

    # Store the original method
    original_method = model._update_causal_mask

    # Set initial values on the model instance
    model.mask_start = start
    model.mask_end = end

    def patched_update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions=False):
        # Call the original method
        causal_mask = original_method(
            attention_mask, input_tensor, cache_position, past_key_values, output_attentions)

        # Check if we should modify the mask
        if (causal_mask is not None and
            hasattr(self, 'mask_start') and hasattr(self, 'mask_end') and
                self.mask_start is not None and self.mask_end is not None):

            start_idx = self.mask_start
            end_idx = self.mask_end

            # Ensure indices are within bounds
            if (causal_mask.dim() >= 4 and
                0 <= start_idx < causal_mask.shape[2] and
                0 <= end_idx < causal_mask.shape[3] and
                    start_idx <= end_idx):

                # Zero out the rectangular region
                # causal_mask shape is typically (batch_size, 1, query_length, key_value_length)
                causal_mask[0, 0, start_idx:end_idx+1, start_idx:end_idx+1] = 0

                logging.info(f"Zeroed out causal mask region [{start_idx}:{
                      end_idx+1}, {start_idx}:{end_idx+1}]")

        return causal_mask

    # Replace the method on the instance
    model._update_causal_mask = types.MethodType(
        patched_update_causal_mask, model)

    logging.info(f"Patched _update_causal_mask on {type(model).__name__}")


def unpatch_causal_mask(model):
    """
    Remove the patch and restore the original _update_causal_mask method.

    Args:
        model: The model instance to unpatch
    """
    # Get the original method from the class
    original_method = type(model)._update_causal_mask

    # Restore it on the instance
    model._update_causal_mask = types.MethodType(original_method, model)

    # Clean up our attributes
    if hasattr(model, 'mask_start'):
        delattr(model, 'mask_start')
    if hasattr(model, 'mask_end'):
        delattr(model, 'mask_end')

    logging.info(f"Unpatched _update_causal_mask on {type(model).__name__}")
