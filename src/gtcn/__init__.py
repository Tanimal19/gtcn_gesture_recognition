import torch.nn as nn
from src.dataset_utils import GestureLabel
from dataclasses import dataclass
from src.gtcn.gcn import GCNLayer, GCNLayerNoPool
from src.gtcn.tcn import (
    TCNLayerLastStep,
    TCNLayerMeanPool,
    TCNLayerWeightPool,
)
from src.gtcn.classifier import (
    RegularClassifier,
    DoubleHeadClassifier,
    ProbThresholdClassifier,
)


@dataclass
class GTCNParams:
    id: str
    WINDOW_LENGTH: int = 15

    GCN_CLASS: str = GCNLayer.__name__
    GCN_DIMS: list[int] = [16]
    GCN_DROPOUT: float = 0.2

    TCN_CLASS: str = TCNLayerLastStep.__name__
    TCN_CHANNELS: list[int] = [64, 64, 64]
    TCN_KERNEL_SIZE: int = 3
    TCN_DILATIONS: list[int] = [1, 2, 4]
    TCN_DROPOUT: float = 0.2

    CLASSIFIER_CLASS: str = RegularClassifier.__name__
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

    OUTPUT_GESTURES = [
        GestureLabel.NONE,
        GestureLabel.GRAB,
        GestureLabel.PINCH,
        GestureLabel.TAP,
        GestureLabel.DENY,
        GestureLabel.KNOB,
        GestureLabel.EXPAND,
    ]

    def __init__(self, hyperparams: GTCNParams):
        super().__init__()

        # === GCN Layer ===
        self.GCNLayer = None
        gcn_features = None
        if hyperparams.GCN_CLASS == GCNLayer.__name__:
            self.GCNLayer = GCNLayer(hyperparams.GCN_DIMS, hyperparams.GCN_DROPOUT)
            gcn_features = hyperparams.GCN_DIMS[-1] * len(GCNLayer.FINGER_GROUPS)
        elif hyperparams.GCN_CLASS == GCNLayerNoPool.__name__:
            self.GCNLayer = GCNLayerNoPool(
                hyperparams.GCN_DIMS, hyperparams.GCN_DROPOUT
            )
            gcn_features = hyperparams.GCN_DIMS[-1] * len(GCNLayer.INPUT_LANDMARKS)

        if self.GCNLayer is None:
            raise ValueError(f"Unsupported GCN class: {hyperparams.GCN_CLASS}")

        # === TCN Layer ===
        self.TCNLayer = None
        gtcn_features = hyperparams.TCN_CHANNELS[-1]
        if hyperparams.TCN_CLASS == TCNLayerLastStep.__name__:
            self.TCNLayer = TCNLayerLastStep(
                [gcn_features] + hyperparams.TCN_CHANNELS,
                hyperparams.TCN_KERNEL_SIZE,
                hyperparams.TCN_DILATIONS,
                hyperparams.TCN_DROPOUT,
            )
        elif hyperparams.TCN_CLASS == TCNLayerMeanPool.__name__:
            self.TCNLayer = TCNLayerMeanPool(
                [gcn_features] + hyperparams.TCN_CHANNELS,
                hyperparams.TCN_KERNEL_SIZE,
                hyperparams.TCN_DILATIONS,
                hyperparams.TCN_DROPOUT,
            )
        elif hyperparams.TCN_CLASS == TCNLayerWeightPool.__name__:
            self.TCNLayer = TCNLayerWeightPool(
                [gcn_features] + hyperparams.TCN_CHANNELS,
                hyperparams.TCN_KERNEL_SIZE,
                hyperparams.TCN_DILATIONS,
                hyperparams.TCN_DROPOUT,
            )

        if self.TCNLayer is None:
            raise ValueError(f"Unsupported TCN class: {hyperparams.TCN_CLASS}")

        # === Classifier ===
        self.Classifier = None
        if hyperparams.CLASSIFIER_CLASS == RegularClassifier.__name__:
            self.Classifier = RegularClassifier(
                gtcn_features,
                len(self.OUTPUT_GESTURES),
                hyperparams.CLASSIFIER_DIM,
            )
        elif hyperparams.CLASSIFIER_CLASS == DoubleHeadClassifier.__name__:
            self.Classifier = DoubleHeadClassifier(
                gtcn_features,
                len(self.OUTPUT_GESTURES) - 1,
                hyperparams.CLASSIFIER_DIM,
                hyperparams.DOUBLE_HEAD_BCE_WEIGHT,
            )
        elif hyperparams.CLASSIFIER_CLASS == ProbThresholdClassifier.__name__:
            self.Classifier = ProbThresholdClassifier(
                gtcn_features,
                len(self.OUTPUT_GESTURES),
                hyperparams.CLASSIFIER_DIM,
                hyperparams.PROB_THRESHOLD,
            )

        if self.GestureClassifier is None:
            raise ValueError(
                f"Unsupported Classifier class: {hyperparams.CLASSIFIER_CLASS}"
            )

    def forward(self, x):
        assert self.GCNLayer is not None
        assert self.TCNLayer is not None
        assert self.Classifier is not None

        B, T, N, C = x.shape

        x_seq = x.reshape(B * T, N, C)  # vectorize over time
        g = self.GCNLayer(x_seq)  # (B*T, gcn_features)
        g = g.reshape(B, T, -1)  # reshape back to (B, T, gcn_features)
        g = g.transpose(1, 2)  # (B, gcn_features, T)
        feat = self.TCNLayer(g)  # (B, gtcn_features)
        return self.Classifier(feat)  # classifier output
