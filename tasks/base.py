from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

from models.base_model_adapter import BaseModelAdapter
from dataset.base_dataset_adapter import BaseDatasetAdapter


class BaseTask(ABC):
    """Abstraction for task-specific example construction and labeling."""

    @abstractmethod
    def build_labels(self, *,
                     example: Dict[str, Any],
                     ds_adapter: BaseDatasetAdapter,
                     model_adapter: BaseModelAdapter,
                     eval_mode: bool) -> Dict[str, Any]:
        """Produce labels and any auxiliary tensors specific to the task."""

    @abstractmethod
    def evaluate_tokens(self,
                        *,
                        example: Dict[str, Any],
                        ds_adapter: BaseDatasetAdapter,
                        model: Any) -> Any:
        """Evaluate the tokens of the model."""

    @abstractmethod
    def evaluate_auxiliary_outputs(self,
                                   *,
                                   example: Dict[str, Any],
                                   ds_adapter: BaseDatasetAdapter,
                                   model: Any) -> Any:
        """Evaluate the auxiliary outputs of the model."""

    @abstractmethod
    def calculate_loss(self, logits, labels, poisson_loss: bool, class_weighting: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for the task."""
