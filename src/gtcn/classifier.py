import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from src.gtcn import OUTPUT_GESTURES
from src.utils import DEVICE

class AbstractClassifier(ABC, nn.Module):
    """
    Abstract base class for classifiers.
    """

    @staticmethod
    @abstractmethod
    def init_criterions(weights: torch.Tensor) -> list[nn.Module]:
        pass

    @staticmethod
    @abstractmethod
    def backpropagate(
        forward_output, y, criterions: list[nn.Module], *args, **kwargs
    ) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def inference(forward_output, *args, **kwargs) -> torch.Tensor:
        pass


class RegularClassifier(AbstractClassifier):
    """
    A classifier that outputs probabilities for all gestures (including NONE).\n
    Input: (B, gtcn_features)\n
    Output: (B, num_gestures)
    """

    def __init__(self, gtcn_features, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gtcn_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(OUTPUT_GESTURES)),
        )

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def init_criterions(weights: torch.Tensor) -> list[nn.Module]:
        entropy_loss_fn = nn.CrossEntropyLoss(weight=weights)
        return [entropy_loss_fn]

    @staticmethod
    def backpropagate(forward_output, y, criterions: list[nn.Module]):
        entropy_loss_fn = criterions[0]
        assert isinstance(entropy_loss_fn, nn.CrossEntropyLoss)

        return entropy_loss_fn(forward_output, y)

    @staticmethod
    def inference(forward_output):
        return torch.argmax(forward_output, dim=1)


class DoubleHeadClassifier(AbstractClassifier):
    """
    A classifier with two heads: one for real gestures and one for NONE likelihood.\n
    Input: (B, gtcn_features)\n
    Outputs:
    - gesture_logits: (B, num_real_gestures)
    - none_logit: (B, 1)
    """

    def __init__(self, gtcn_features, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(gtcn_features, hidden_dim),
            nn.ReLU(),
        )
        self.gesture_head = nn.Linear(hidden_dim, len(OUTPUT_GESTURES) - 1)
        self.none_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        gesture_logits = self.gesture_head(h)
        none_logit = self.none_head(h)
        return gesture_logits, none_logit

    @staticmethod
    def init_criterions(weights: torch.Tensor) -> list[nn.Module]:
        entropy_loss_fn = nn.CrossEntropyLoss(weight=weights[1:])  # exclude NONE weight
        bce_loss_fn = nn.BCEWithLogitsLoss()
        return [entropy_loss_fn, bce_loss_fn]

    @staticmethod
    def backpropagate(
        forward_output, y, criterions: list[nn.Module], bce_weight: float
    ):
        gesture_logits, none_logit = forward_output
        entropy_loss_fn = criterions[0]
        bce_loss_fn = criterions[1]
        assert isinstance(entropy_loss_fn, nn.CrossEntropyLoss)
        assert isinstance(bce_loss_fn, nn.BCEWithLogitsLoss)

        gesture_labels = y.clone()
        real_gesture_mask = y != 0
        if real_gesture_mask.any():
            gesture_loss = entropy_loss_fn(
                gesture_logits[real_gesture_mask],
                gesture_labels[real_gesture_mask]
                - 1,  # shift real gesture labels to start from 0
            )
        else:
            gesture_loss = torch.tensor(0.0, device=gesture_logits.device)

        none_labels = (y == 0).float()  # 1: none gesture, 0: real gesture
        none_loss = bce_loss_fn(none_logit.squeeze(), none_labels)

        return gesture_loss + bce_weight * none_loss

    @staticmethod
    def inference(forward_output):
        gesture_logits, none_logit = forward_output
        none_prob = torch.sigmoid(none_logit).squeeze(-1)
        pred_y = torch.where(
            none_prob > 0.5,
            torch.zeros(none_prob.shape, dtype=torch.long, device=DEVICE),  # NONE
            torch.argmax(gesture_logits, dim=1) + 1,  # shift by 1 to account for NONE
        )
        return pred_y


class ProbThresholdClassifier(AbstractClassifier):
    """
    A classifier that outputs probabilities only for real gestures. Determine NONE if all probability less than threshold.\n
    Input: (B, gtcn_features)\n
    Outputs: (B, num_real_gestures)
    """

    def __init__(self, gtcn_features, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(gtcn_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(OUTPUT_GESTURES) - 1),
        )

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def init_criterions(weights: torch.Tensor) -> list[nn.Module]:
        entropy_loss_fn = nn.CrossEntropyLoss(weight=weights[1:])  # exclude NONE weight
        return [entropy_loss_fn]

    @staticmethod
    def backpropagate(forward_output, y, criterions: list[nn.Module]):
        entropy_loss_fn = criterions[0]
        assert isinstance(entropy_loss_fn, nn.CrossEntropyLoss)

        logits = forward_output
        real_gesture_mask = y != 0
        if real_gesture_mask.any():
            return entropy_loss_fn(
                logits[real_gesture_mask],
                y[real_gesture_mask] - 1,  # shift real gesture labels to start from 0
            )
        else:
            return logits.sum() * 0.0

    @staticmethod
    def inference(forward_output, none_threshold):
        logits = forward_output
        probs = torch.softmax(logits, dim=1)
        max_probs, max_indices = torch.max(probs, dim=1)

        pred_y = torch.where(
            max_probs < none_threshold,
            torch.zeros_like(max_indices, dtype=torch.long, device=DEVICE),  # NONE
            max_indices + 1,  # shift by 1 to account for NONE
        )
        return pred_y
