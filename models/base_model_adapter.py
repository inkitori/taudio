from torch import nn
import torch
from contextlib import contextmanager
from typing import Any


class BaseModelAdapter(nn.Module):
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

    @property
    def processor(self) -> Any:
        raise NotImplementedError

    @property
    def seconds_to_embedding(self) -> int:
        raise NotImplementedError
        # Core calls
    # Core calls

    @property
    def text_model(self) -> nn.Module:
        raise NotImplementedError

    def forward(self, **kwargs) -> Any:
        raise NotImplementedError

    def generate(self, **kwargs):
        raise NotImplementedError

    # Model-specific helpers
    @contextmanager
    def bidirectional_audio_context(self, input_ids: torch.Tensor):
        """Default no-op context; adapters override to implement model-specific patching."""
        yield
