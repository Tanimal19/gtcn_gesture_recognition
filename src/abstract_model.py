from dataset_utils import GestureLabel, HandLandmark
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn


class AbstractGestureModel(ABC, nn.Module):
    WINDOW_LENGTH: int

    @abstractmethod
    def landmarks_window_to_X(self, landmarks_window: np.ndarray) -> torch.Tensor:
        """
        landmarks_window: np.array of shape (WINDOW_LENGTH, len(HandLandmark), 3)\n
        transform raw landmarks window to model required feature representation.
        """
        pass

    @abstractmethod
    def y_to_label(self, y: int) -> GestureLabel:
        """
        map model output y to GestureLabel
        """
        pass
