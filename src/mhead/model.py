import torch
import torch.nn as nn
import torch.nn.functional as F
from src.gtcn.model import generate_adjacent_matrix, GTCNModelParams
from src.dataset_utils import GestureLabel, HandLandmark


class GTCNMHead(nn.Module):
    """
    Gesture Recognition Network\n
    Input: (B, window_length, num_landmarks, 3)\n
    Output: (B, num_gestures)
    """

    WINDOW_LENGTH = 10
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
    GESTURES = [  # none + fine dynamic gestures
        GestureLabel.NONE,
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

        def __init__(self, hidden_dim, dropout):
            super().__init__()
            in_dim = 3
            out_dim = hidden_dim

            self.W1 = nn.Linear(in_dim, out_dim, bias=False)
            self.W2 = nn.Linear(in_dim, out_dim, bias=False)

            # residual projection
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False)

            # dropout
            self.dropout = nn.Dropout(dropout)

            # adjacency matrices (fixed)
            A1 = torch.tensor(
                generate_adjacent_matrix(
                    GTCNMHead.LANDMARKS, self.INSIDE_FINGER_CONNECTIONS
                ),
                dtype=torch.float32,
            )
            A2 = torch.tensor(
                generate_adjacent_matrix(
                    GTCNMHead.LANDMARKS, self.BETWEEN_FINGER_CONNECTIONS
                ),
                dtype=torch.float32,
            )
            self.register_buffer("A1", A1)
            self.register_buffer("A2", A2)

        def forward(self, X):
            AX1 = torch.matmul(self.A1, X)
            AX2 = torch.matmul(self.A2, X)

            out = self.W1(AX1) + self.W2(AX2)

            res = self.res_proj(X)

            out = F.relu(out + res)
            out = self.dropout(out)

            return out

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

    class TCNBlock(nn.Module):
        """
        Temporal Convolutional Network Block\n
        Input: (B, in_ch, T)\n
        Output: (B, out_ch, T)
        """

        def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
            super().__init__()
            pad = dilation * (kernel_size - 1)

            self.conv = nn.Conv1d(
                in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=0
            )

            self.padding = pad

            self.dropout = nn.Dropout(dropout)

            self.res_proj = None
            if in_ch != out_ch:
                self.res_proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            res = x

            # manual right padding
            x = F.pad(x, (0, self.padding))

            out = self.conv(x)
            out = F.relu(out)
            out = self.dropout(out)

            if self.res_proj is not None:
                res = self.res_proj(res)

            return F.relu(out + res)

    class TemporalConvNet(nn.Module):
        """
        Temporal Convolutional Network (multiple TCN blocks)\n
        Input: (B, gcn_hidden_dim*3, T)\n
        Output: (B, tcn_hidden_dim)
        """

        def __init__(self, gcn_hidden_dim, hidden_dim, kernel_size, dilations, dropout):
            super().__init__()
            in_ch = gcn_hidden_dim * 3
            layers = []

            for d in dilations:
                layers.append(
                    GTCNMHead.TCNBlock(
                        in_ch,
                        hidden_dim,
                        kernel_size=kernel_size,
                        dilation=d,
                        dropout=dropout,
                    )
                )
                in_ch = hidden_dim

            self.layers = nn.ModuleList(layers)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x.mean(dim=2)  # global average pooling

    class GestureMheadClassifier(nn.Module):
        """
        Gesture Classifier with:
        - gesture_head: logits for real gestures (exclude NONE)
        - none_head: a single logit indicating NONE likelihood
        """

        def __init__(self, tcn_hidden_dim, hidden_dim, num_real_gestures):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(tcn_hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.gesture_head = nn.Linear(hidden_dim, num_real_gestures)
            self.none_head = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            h = self.shared(x)
            gesture_logits = self.gesture_head(h)  # (B, num_real_gestures)
            none_logit = self.none_head(h).squeeze(-1)  # (B,)
            return gesture_logits, none_logit

    def __init__(self, hyperparams: GTCNModelParams):
        super().__init__()

        self.gcn = self.GCNLayer(hyperparams.GCN_HIDDEN_DIM, hyperparams.GCN_DROPOUT)
        self.pool = self.FingerPooling()
        self.tcn = self.TemporalConvNet(
            hyperparams.GCN_HIDDEN_DIM,
            hyperparams.TCN_HIDDEN_DIM,
            hyperparams.TCN_KERNEL_SIZE,
            hyperparams.TCN_DILATIONS,
            hyperparams.TCN_DROPOUT,
        )
        self.classifier = self.GestureMheadClassifier(
            hyperparams.TCN_HIDDEN_DIM,
            hyperparams.CLASS_HIDDEN_DIM,
            len(self.GESTURES) - 1,  # exclude NONE
        )

    def forward(self, x):
        B, T, N, C = x.shape

        x_seq = x.reshape(B * T, N, C)  # vectorize over time
        g = self.gcn(x_seq)  # (B*T, num_landmarks, gcn_hidden_dim)
        g = self.pool(g)  # (B*T, 3, gcn_hidden_dim)
        g = g.reshape(B, T, -1)  # (B, T, gcn_hidden_dim*3)
        g = g.transpose(1, 2)  # (B, gcn_hidden_dim*3, T)
        feat = self.tcn(g)  # (B, tcn_hidden_dim)
        gesture_logits, none_logit = self.classifier(feat)

        return gesture_logits, none_logit
