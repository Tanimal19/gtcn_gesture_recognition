import torch
import torch.nn as nn
from src.gtcn.gcn import BaseGCNLayer, GCNLayerFingerPool, GCNLayerNoPool
from src.gtcn.tcn import BaseTCNLayer
from gtcn.classifier import (
    AbstractClassifier,
    RegularClassifier,
    DoubleHeadClassifier,
    ProbThresholdClassifier,
)
from dataclasses import dataclass
from typing import Type


@dataclass
class GTCNParams:
    id: str
    GCN_CLASS: Type[BaseGCNLayer]
    TCN_CLASS: Type[BaseTCNLayer]
    CLASSIFIER_CLASS: Type[AbstractClassifier]

    WINDOW_LENGTH: int = 15

    GCN_DIMS: list[int] = [16]
    GCN_DROPOUT: float = 0.2

    TCN_CHANNELS: list[int] = [64, 64, 64]
    TCN_KERNEL_SIZE: int = 3
    TCN_DILATIONS: list[int] = [1, 2, 4]
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

        self.gcn_layer = hyperparams.GCN_CLASS(
            hyperparams.GCN_DIMS, hyperparams.GCN_DROPOUT
        )

        gcn_feature_dim = self.gcn_layer.get_outdim()
        self.tcn_layer = hyperparams.TCN_CLASS(
            [gcn_feature_dim] + hyperparams.TCN_CHANNELS,
            hyperparams.TCN_KERNEL_SIZE,
            hyperparams.TCN_DILATIONS,
            hyperparams.TCN_DROPOUT,
        )

        gtcn_features = self.tcn_layer.get_outdim()
        if hyperparams.CLASSIFIER_CLASS == DoubleHeadClassifier:
            self.classifier = DoubleHeadClassifier(
                gtcn_features,
                hyperparams.CLASSIFIER_DIM,
                hyperparams.DOUBLE_HEAD_BCE_WEIGHT,
            )
        elif hyperparams.CLASSIFIER_CLASS == ProbThresholdClassifier:
            self.classifier = ProbThresholdClassifier(
                gtcn_features,
                hyperparams.CLASSIFIER_DIM,
                hyperparams.PROB_THRESHOLD,
            )
        else:  # RegularClassifier
            self.classifier = RegularClassifier(
                gtcn_features,
                hyperparams.CLASSIFIER_DIM,
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

        # Classifier: predict gesture
        return self.classifier(feat)
