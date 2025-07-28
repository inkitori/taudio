import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def visualize_causal_mask(causal_mask: torch.Tensor,
                          batch_idx: int = 0,
                          head_idx: int = 0,
                          max_size: int = 100,
                          title: str = "Causal Mask",
                          save_path: Optional[str] = None):
    """
    Visualize a causal mask as a heatmap.

    Args:
        causal_mask: The causal mask tensor (batch_size, num_heads, seq_len, seq_len)
        batch_idx: Which batch element to visualize
        head_idx: Which attention head to visualize  
        max_size: Maximum size to display (for large sequences)
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    if causal_mask is None:
        print("Causal mask is None")
        return

    if causal_mask.dim() < 4:
        print(f"Expected 4D mask, got {causal_mask.dim()}D")
        return

    # Extract the specific batch and head
    mask_2d = causal_mask[batch_idx, head_idx].cpu().numpy()

    # Truncate if too large
    if mask_2d.shape[0] > max_size:
        mask_2d = mask_2d[:max_size, :max_size]
        title += f" (truncated to {max_size}x{max_size})"

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(mask_2d, cmap='RdYlBu_r', aspect='auto')
    plt.colorbar(label='Mask Value')
    plt.title(f"{title}\nBatch {batch_idx}, Head {
              head_idx}, Shape: {mask_2d.shape}")
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def compare_masks(original_mask: torch.Tensor,
                  modified_mask: torch.Tensor,
                  batch_idx: int = 0,
                  head_idx: int = 0,
                  max_size: int = 100,
                  save_path: Optional[str] = None):
    """
    Compare original and modified causal masks side by side.

    Args:
        original_mask: The original causal mask
        modified_mask: The modified causal mask
        batch_idx: Which batch element to visualize
        head_idx: Which attention head to visualize
        max_size: Maximum size to display
        save_path: Optional path to save the comparison
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    if original_mask is not None:
        orig_2d = original_mask[batch_idx, head_idx].cpu().numpy()
        if orig_2d.shape[0] > max_size:
            orig_2d = orig_2d[:max_size, :max_size]

        im1 = ax1.imshow(orig_2d, cmap='RdYlBu_r', aspect='auto')
        ax1.set_title('Original Mask')
        ax1.set_xlabel('Key Position')
        ax1.set_ylabel('Query Position')
        plt.colorbar(im1, ax=ax1)
    else:
        ax1.text(0.5, 0.5, 'Original mask is None',
                 ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Original Mask')

    if modified_mask is not None:
        mod_2d = modified_mask[batch_idx, head_idx].cpu().numpy()
        if mod_2d.shape[0] > max_size:
            mod_2d = mod_2d[:max_size, :max_size]

        im2 = ax2.imshow(mod_2d, cmap='RdYlBu_r', aspect='auto')
        ax2.set_title('Modified Mask')
        ax2.set_xlabel('Key Position')
        ax2.set_ylabel('Query Position')
        plt.colorbar(im2, ax=ax2)

        # Show difference if both masks exist
        if original_mask is not None:
            diff = mod_2d - orig_2d
            im3 = ax3.imshow(diff, cmap='RdBu', aspect='auto')
            ax3.set_title('Difference (Modified - Original)')
            ax3.set_xlabel('Key Position')
            ax3.set_ylabel('Query Position')
            plt.colorbar(im3, ax=ax3)
        else:
            ax3.axis('off')
    else:
        ax2.text(0.5, 0.5, 'Modified mask is None',
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Modified Mask')
        ax3.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()


def create_mask_inspection_patch(model):
    """
    Create a patch that captures and stores causal masks for inspection.

    Args:
        model: The model to patch

    Returns:
        A function to retrieve captured masks
    """
    import types

    # Store the original method
    original_method = model._update_causal_mask

    # Storage for captured masks
    model._captured_masks = []

    def patched_update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions=False):
        # Call the original method
        causal_mask = original_method(
            attention_mask, input_tensor, cache_position, past_key_values, output_attentions)

        # Store a copy of the mask for inspection
        if causal_mask is not None:
            self._captured_masks.append({
                'mask': causal_mask.clone().detach(),
                'shape': causal_mask.shape,
                'call_count': len(self._captured_masks) + 1
            })
        else:
            self._captured_masks.append({
                'mask': None,
                'shape': None,
                'call_count': len(self._captured_masks) + 1
            })

        return causal_mask

    # Replace the method
    model._update_causal_mask = types.MethodType(
        patched_update_causal_mask, model)

    def get_captured_masks():
        """Return all captured masks"""
        return model._captured_masks

    def clear_captured_masks():
        """Clear the captured masks"""
        model._captured_masks = []

    def get_latest_mask():
        """Get the most recently captured mask"""
        if model._captured_masks:
            return model._captured_masks[-1]['mask']
        return None

    return get_captured_masks, clear_captured_masks, get_latest_mask


# Example usage function
def inspect_mask_changes(model, inputs, patch_function, *patch_args, **patch_kwargs):
    """
    Helper function to inspect how a patch changes the causal mask.

    Args:
        model: The model to test
        inputs: Input data for the model
        patch_function: The patching function to apply
        *patch_args, **patch_kwargs: Arguments for the patch function
    """

    # First, capture the original mask
    get_masks, clear_masks, get_latest = create_mask_inspection_patch(model)
    clear_masks()

    print("Running with original mask...")
    _ = model(**inputs)
    original_mask = get_latest()
    print(f"Original mask shape: {
          original_mask.shape if original_mask is not None else 'None'}")

    # Apply the patch and capture the modified mask
    clear_masks()
    patch_function(model, *patch_args, **patch_kwargs)

    print("Running with patched mask...")
    _ = model(**inputs)
    modified_mask = get_latest()
    print(f"Modified mask shape: {
          modified_mask.shape if modified_mask is not None else 'None'}")

    # Compare the masks
    if original_mask is not None and modified_mask is not None:
        print("Visualizing mask comparison...")
        compare_masks(original_mask, modified_mask,
                      save_path='mask_comparison.png')

        # Calculate some statistics
        diff = (modified_mask - original_mask).abs()
        num_changed = (diff > 1e-6).sum().item()
        total_elements = diff.numel()
        print(f"Changed elements: {
              num_changed}/{total_elements} ({100*num_changed/total_elements:.2f}%)")
    else:
        print("Cannot compare masks - one or both are None")


if __name__ == "__main__":
    # Example of how to use the visualization tools
    print("Mask visualization utilities loaded.")
    print("Use visualize_causal_mask() to visualize a single mask")
    print("Use compare_masks() to compare two masks")
    print("Use inspect_mask_changes() to see how patches affect masks")
