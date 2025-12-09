import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.dataset_utils import GestureLabel, HandLandmark


def generate_adjacent_matrix(
    landmarks: list[HandLandmark],
    connections: list[tuple[HandLandmark, HandLandmark]],
) -> np.ndarray:
    N = len(landmarks)
    adj = np.zeros((N, N), dtype=int)

    node2idx = {lm.name: idx for idx, lm in enumerate(landmarks)}
    for a, b in connections:
        i, j = node2idx[a.name], node2idx[b.name]
        adj[i, j] = 1
        adj[j, i] = 1

    return adj


class GTCNModel(nn.Module):
    """
    Gesture Recognition Network\n
    Input: (B, window_length, num_landmarks, 3)\n
    Output: (B, num_gestures)
    """

    WINDOW_LENGTH = 20
    GCN_HIDDEN_DIM = 16  # GCN hidden dimension
    TCN_HIDDEN_DIM = 64  # TCN hidden dimension
    CLASS_HIDDEN_DIM = 32  # Classifier hidden dimension
    LANDMARKS = [
        HandLandmark.thumbA,
        HandLandmark.thumbB,
        HandLandmark.thumbEnd,
        HandLandmark.indexA,
        HandLandmark.indexB,
        HandLandmark.indexC,
        HandLandmark.indexEnd,
        HandLandmark.middleA,
        HandLandmark.middleB,
        HandLandmark.middleC,
        HandLandmark.middleEnd,
    ]
    GESTURES = [  # none + dynamic gestures
        GestureLabel.NONE,
        GestureLabel.LEFT,
        GestureLabel.RIGHT,
        GestureLabel.CIRCLE,
        GestureLabel.V,
        GestureLabel.CROSS,
        GestureLabel.GRAB,
        GestureLabel.PINCH,
        GestureLabel.TAP,
        GestureLabel.DENY,
        GestureLabel.KNOB,
        GestureLabel.EXPAND,
    ]

    class GCNLayer(nn.Module):
        """
        Graph Convolution Layer\n
        Input: (num_landmarks, 3)\n
        Output: (num_landmarks, gcn_hidden_dim)
        """

        INSIDE_FINGER_CONNECTIONS = [
            (HandLandmark.thumbA, HandLandmark.thumbB),
            (HandLandmark.thumbB, HandLandmark.thumbEnd),
            (HandLandmark.indexA, HandLandmark.indexB),
            (HandLandmark.indexB, HandLandmark.indexC),
            (HandLandmark.indexC, HandLandmark.indexEnd),
            (HandLandmark.middleA, HandLandmark.middleB),
            (HandLandmark.middleB, HandLandmark.middleC),
            (HandLandmark.middleC, HandLandmark.middleEnd),
        ]
        BETWEEN_FINGER_CONNECTIONS = [
            (HandLandmark.thumbEnd, HandLandmark.indexEnd),
            (HandLandmark.thumbEnd, HandLandmark.middleEnd),
            (HandLandmark.indexEnd, HandLandmark.middleEnd),
        ]

        def __init__(self, hidden_dim):
            super().__init__()
            in_dim = 3
            out_dim = hidden_dim

            self.W1 = nn.Linear(in_dim, out_dim, bias=False)
            self.W2 = nn.Linear(in_dim, out_dim, bias=False)

            # adjacency matrices (fixed)
            A1 = torch.tensor(
                generate_adjacent_matrix(
                    GTCNModel.LANDMARKS, self.INSIDE_FINGER_CONNECTIONS
                ),
                dtype=torch.float32,
            )
            A2 = torch.tensor(
                generate_adjacent_matrix(
                    GTCNModel.LANDMARKS, self.BETWEEN_FINGER_CONNECTIONS
                ),
                dtype=torch.float32,
            )
            self.register_buffer("A1", A1)
            self.register_buffer("A2", A2)

        def forward(self, X):
            AX1 = torch.matmul(self.A1, X)
            AX2 = torch.matmul(self.A2, X)

            out = self.W1(AX1) + self.W2(AX2)
            return F.relu(out)

    class FingerPooling(nn.Module):
        """
        Finger-group Average Pooling.\n
        Input: (num_landmarks, gcn_hidden_dim)\n
        Output: (num_fingers, gcn_hidden_dim)
        """

        FINGER_GROUPS = [
            [0, 1, 2],  # thumb
            [3, 4, 5, 6],  # index finger
            [7, 8, 9, 10],  # middle finger
        ]

        def __init__(self):
            super().__init__()

        def forward(self, X):
            pooled = []
            for idxs in self.FINGER_GROUPS:
                pooled.append(X[:, idxs].mean(dim=1))
            return torch.stack(pooled, dim=1)

    class TemporalConvNet(nn.Module):
        """
        Temporal Convolution Network\n
        Input: (B, gcn_hidden_dim*3, window_length)\n
        Output: (B, tcn_hidden_dim) after GAP
        """

        KERNEL_SIZE = 3
        DILATIONS = [1, 3, 9, 27]

        def __init__(self, hidden_dim):
            super().__init__()
            in_ch = GTCNModel.GCN_HIDDEN_DIM * 3
            out_ch = hidden_dim

            self.padding = [d * (self.KERNEL_SIZE - 1) for d in self.DILATIONS]

            layers = []
            for d in self.DILATIONS:
                layers.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_ch,
                            out_ch,
                            kernel_size=self.KERNEL_SIZE,
                            padding=0,  # we will pad manually
                            dilation=d,
                        ),
                        nn.ReLU(),
                    )
                )
                in_ch = out_ch

            self.layers = nn.ModuleList(layers)

        def forward(self, x):
            for conv, p in zip(self.layers, self.padding):
                x = F.pad(x, (0, p))  # pad at the end
                x = conv(x)

            # Global Average Pooling (over window_length)
            x = x.mean(dim=2)
            return x

    class GestureClassifier(nn.Module):
        """
        Gesture Classifier (outputs logits for CrossEntropyLoss)\n
        Input: (B, tcn_hidden_dim)\n
        Output: (B, num_gestures)
        """

        def __init__(self, hidden_dim, num_gestures):
            super().__init__()

            in_dim = GTCNModel.TCN_HIDDEN_DIM

            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_gestures),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self):
        super().__init__()

        self.gcn = self.GCNLayer(self.GCN_HIDDEN_DIM)
        self.pool = self.FingerPooling()
        self.tcn = self.TemporalConvNet(self.TCN_HIDDEN_DIM)
        self.classifier = self.GestureClassifier(
            self.CLASS_HIDDEN_DIM, len(self.GESTURES)
        )

    def forward(self, x):
        B, T, N, C = x.shape

        x_seq = x.reshape(B * T, N, C)  # vectorize over time
        g = self.gcn(x_seq)  # (B*T, num_landmarks, gcn_hidden_dim)
        g = self.pool(g)  # (B*T, 3, gcn_hidden_dim)
        g = g.reshape(B, T, -1)  # (B, T, gcn_hidden_dim*3)
        g = g.transpose(1, 2)  # (B, gcn_hidden_dim*3, T)
        feat = self.tcn(g)  # (B, tcn_hidden_dim)
        logits = self.classifier(feat)  # (B, num_gestures)

        return logits
