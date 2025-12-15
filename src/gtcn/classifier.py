import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractClassifier(ABC, nn.Module):
    """
    Abstract base class for classifiers.
    """

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backpropagate(self, forward_output, y, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def inference(self, forward_output) -> torch.Tensor:
        pass


class RegularClassifier(AbstractClassifier):
    """
    A classifier that outputs probabilities for all gestures (including NONE).\n
    Input: (B, gtcn_features)\n
    Output: (B, num_gestures)
    """

    def __init__(self, gtcn_features, num_gestures, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gtcn_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_gestures),
        )

    def forward(self, x):
        return self.net(x)

    def backpropagate(self, forward_output, y, entropy_loss_fn):
        return entropy_loss_fn(forward_output, y)

    def inference(self, forward_output):
        return torch.argmax(forward_output, dim=1)


class DoubleHeadClassifier(nn.Module):
    """
    A classifier with two heads: one for real gestures and one for NONE likelihood.\n
    Input: (B, gtcn_features)\n
    Outputs:
    - gesture_logits: (B, num_real_gestures)
    - none_logit: (B, 1)
    """

    def __init__(self, gtcn_features, num_real_gestures, hidden_dim, bce_weight=0.1):
        super().__init__()
        self.num_real_gestures = num_real_gestures
        self.bce_weight = bce_weight

        self.shared = nn.Sequential(
            nn.Linear(gtcn_features, hidden_dim),
            nn.ReLU(),
        )
        self.gesture_head = nn.Linear(hidden_dim, num_real_gestures)
        self.none_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        gesture_logits = self.gesture_head(h)
        none_logit = self.none_head(h)
        return gesture_logits, none_logit

    def backpropagate(self, forward_output, y, entropy_loss_fn, bce_loss_fn):
        gesture_logits, none_logit = forward_output
        gesture_labels = y.clone()
        real_gesture_mask = y != -1
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

        return gesture_loss + self.bce_weight * none_loss

    def inference(self, forward_output):
        gesture_logits, none_logit = forward_output
        none_prob = torch.sigmoid(none_logit)
        pred_y = torch.where(
            none_prob > 0.5,
            torch.zeros_like(none_logit, dtype=torch.long),  # NONE
            torch.argmax(gesture_logits, dim=1) + 1,  # shift by 1 to account for NONE
        )
        return pred_y


class ProbThresholdClassifier(nn.Module):
    """
    A classifier that outputs probabilities only for real gestures. Determine NONE if all probability less than threshold.\n
    Input: (B, gtcn_features)\n
    Outputs: (B, num_real_gestures)
    """

    def __init__(
        self, gtcn_features, num_real_gestures, hidden_dim, none_threshold=0.6
    ):
        super().__init__()
        self.none_threshold = none_threshold

        self.net = nn.Sequential(
            nn.Linear(gtcn_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_real_gestures),
        )

    def forward(self, x):
        return self.net(x)

    def backpropagate(self, forward_output, y, entropy_loss_fn):
        logits = forward_output
        real_gesture_mask = y != -1
        if real_gesture_mask.any():
            return entropy_loss_fn(
                logits[real_gesture_mask],
                y[real_gesture_mask] - 1,  # shift real gesture labels to start from 0
            )
        else:
            return torch.tensor(0.0, device=logits.device)

    def inference(self, forward_output):
        logits = forward_output
        probs = torch.softmax(logits, dim=1)
        max_probs, max_indices = torch.max(probs, dim=1)

        pred_y = torch.where(
            max_probs < self.none_threshold,
            torch.zeros_like(max_indices, dtype=torch.long),  # NONE
            max_indices + 1,  # shift by 1 to account for NONE
        )
        return pred_y
