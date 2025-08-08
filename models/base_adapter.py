from torch import nn
import torch
from contextlib import contextmanager
from typing import Any


class BaseAudioTextAdapter(nn.Module):
    """
    Minimal interface that TAudio relies on, implemented by concrete backends.
    """

    def __init__(self) -> None:
        super().__init__()

    # Construction helpers
    @property
    def hidden_dim(self) -> int:  # output dim of audio encoder hidden states
        raise NotImplementedError

    @property
    def dtype(self) -> torch.dtype:
        raise NotImplementedError

    @property
    def audio_id(self) -> int:
        raise NotImplementedError

    @property
    def assistant_id(self) -> int:
        raise NotImplementedError

    # Core calls
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        labels: torch.Tensor,
        output_hidden_states: bool,
    ) -> Any:
        raise NotImplementedError

    def generate(self, **kwargs):
        raise NotImplementedError

    # Model-specific helpers
    def get_audio_bounds(self, input_ids: torch.Tensor) -> tuple[int, int]:
        """Return [start_index, end_index] for the audio span in input_ids."""
        raise NotImplementedError

    @contextmanager
    def bidirectional_audio_context(self, input_ids: torch.Tensor):
        """Default no-op context; adapters override to implement model-specific patching."""
        yield
