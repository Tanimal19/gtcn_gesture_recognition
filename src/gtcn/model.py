import torch.nn as nn
from src.gtcn.gcn import BaseGCNLayer
from src.gtcn.tcn import BaseTCNLayer
from src.gtcn.classifier import (
    AbstractClassifier,
    RegularClassifier,
    DoubleHeadClassifier,
    ProbThresholdClassifier,
)
from dataclasses import dataclass, field
from typing import Type
import copy


@dataclass
class GTCNParams:
    id: str
    GCN_CLASS: Type[BaseGCNLayer]
    TCN_CLASS: Type[BaseTCNLayer]
    CLASSIFIER_CLASS: Type[AbstractClassifier]

    WINDOW_LENGTH: int = 15

    GCN_DIMS: list[int] = field(default_factory=lambda: [16])
    GCN_DROPOUT: float = 0.2

    TCN_CHANNELS: list[int] = field(default_factory=lambda: [64, 64, 64])
    TCN_KERNEL_SIZE: int = 3
    TCN_DILATIONS: list[int] = field(default_factory=lambda: [1, 2, 4])
    TCN_DROPOUT: float = 0.2

    CLASSIFIER_DIM = 32
    DOUBLE_HEAD_BCE_WEIGHT: float = 0.1
    PROB_THRESHOLD: float = 0.6


class GTCNModel(nn.Module):
    """
    GCNLayer + TCNLayer + Classifier
    - input: (B, window_length, num_landmarks=11, num_dimensions=3)
    - GCNLayer output: (B, window_length, gcn_features)
    - TCNLayer output: (B, gtcn_features)
    - Classifier output: please refer to :class:`AbstractClassifier`
    """

    def __init__(self, hyperparams: GTCNParams):
        super().__init__()

        self.hyperparams = copy.deepcopy(hyperparams)

        self.gcn_layer = self.hyperparams.GCN_CLASS(
            self.hyperparams.GCN_DIMS, self.hyperparams.GCN_DROPOUT
        )

        gcn_feature_dim = self.gcn_layer.get_outdim()
        self.tcn_layer = self.hyperparams.TCN_CLASS(
            [gcn_feature_dim] + self.hyperparams.TCN_CHANNELS,
            self.hyperparams.TCN_KERNEL_SIZE,
            self.hyperparams.TCN_DILATIONS,
            self.hyperparams.TCN_DROPOUT,
        )

        gtcn_features = self.tcn_layer.get_outdim()
        self.classifier = self.hyperparams.CLASSIFIER_CLASS(
            gtcn_features, self.hyperparams.CLASSIFIER_DIM
        )

    def forward(self, x):
        B, T, N, C = x.shape

        # GCN: process spatial features
        x_seq = x.reshape(B * T, N, C)  # vectorize over time
        g = self.gcn_layer(x_seq)  # (B*T, gcn_features)

        # TCN: process temporal features
        g = g.reshape(B, T, -1)  # reshape back to (B, T, gcn_features)
        g = g.transpose(1, 2)  # (B, gcn_features, T)
        feat = self.tcn_layer(g)  # (B, gtcn_features)

        # Classifier
        return self.classifier(feat)

    def compute_loss(self, forward_output, y, criterions: list[nn.Module]):
        if isinstance(self.classifier, DoubleHeadClassifier):
            return self.classifier.backpropagate(
                forward_output,
                y,
                criterions,
                self.hyperparams.DOUBLE_HEAD_BCE_WEIGHT,
            )
        else:
            return self.classifier.backpropagate(forward_output, y, criterions)

    def inference_gesture(self, forward_output):
        if isinstance(self.classifier, ProbThresholdClassifier):
            return self.classifier.inference(
                forward_output, self.hyperparams.PROB_THRESHOLD
            )
        else:
            return self.classifier.inference(forward_output)
