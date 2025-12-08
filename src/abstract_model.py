from src.dataset_utils import GestureLabel
from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn


class AbstractGestureModel(ABC, nn.Module):
    @staticmethod
    @abstractmethod
    def predict_label(model: nn.Module, x: torch.Tensor, *args, **kwargs) -> Any:
        """Predict y from input tensor x using the model. Should be used combined with `convert_y_to_gesture()`."""

    @staticmethod
    @abstractmethod
    def convert_y_to_gesture(y) -> GestureLabel | None:
        """Map model output integer label to GestureLabel enum."""
        pass
