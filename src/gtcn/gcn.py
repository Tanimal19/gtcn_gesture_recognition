import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.dataset_utils import HandLandmark


class GCNLayer(nn.Module):
    """
    Stacked GCN Blocks with finger pooling at the end\n
    Input: (num_landmarks=11, num_dimensions=3)\n
    Output: (gcn_features*num_finger_groups)
    """

    INPUT_LANDMARKS = [
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
    INPUT_DIMENSIONS = 3  # x, y, z

    INTRA_FINGER_CONNECTIONS = [
        (HandLandmark.thumbA, HandLandmark.thumbB),
        (HandLandmark.thumbB, HandLandmark.thumbEnd),
        (HandLandmark.indexA, HandLandmark.indexB),
        (HandLandmark.indexB, HandLandmark.indexC),
        (HandLandmark.indexC, HandLandmark.indexEnd),
        (HandLandmark.middleA, HandLandmark.middleB),
        (HandLandmark.middleB, HandLandmark.middleC),
        (HandLandmark.middleC, HandLandmark.middleEnd),
    ]
    INTER_FINGER_CONNECTIONS = [
        (HandLandmark.thumbEnd, HandLandmark.indexEnd),
        (HandLandmark.thumbEnd, HandLandmark.middleEnd),
        (HandLandmark.indexEnd, HandLandmark.middleEnd),
    ]

    FINGER_GROUPS = [
        [0, 1, 2],  # thumb
        [3, 4, 5, 6],  # index finger
        [7, 8, 9, 10],  # middle finger
    ]

    def __init__(self, hidden_dims: list[int], dropout):
        super().__init__()

        A1 = self._normalize_adjacency(
            torch.tensor(
                self._generate_adjacent_matrix(
                    self.INPUT_LANDMARKS, self.INTRA_FINGER_CONNECTIONS
                ),
                dtype=torch.float32,
            )
        )
        A2 = self._normalize_adjacency(
            torch.tensor(
                self._generate_adjacent_matrix(
                    self.INPUT_LANDMARKS, self.INTER_FINGER_CONNECTIONS
                ),
                dtype=torch.float32,
            )
        )
        self.register_buffer("A1", A1)
        self.register_buffer("A2", A2)

        hidden_dims.insert(0, self.INPUT_DIMENSIONS)
        gcn_blocks = []
        for i in range(len(hidden_dims) - 1):
            gcn_blocks.append(
                GCNBlock(
                    in_dim=hidden_dims[i],
                    out_dim=hidden_dims[i + 1],
                    dropout=dropout,
                    A1=self.A1,
                    A2=self.A2,
                )
            )
        self.net = nn.Sequential(*gcn_blocks)

    def forward(self, X):
        out = self.net(X)

        pooled = []
        for idxs in self.FINGER_GROUPS:
            pooled.append(out[:, idxs].mean(dim=1))
        out = torch.stack(pooled, dim=1)

        out = out.reshape(out.size(0), -1)
        return out

    @staticmethod
    def _generate_adjacent_matrix(
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

    @staticmethod
    def _normalize_adjacency(A, add_self_loop=True):
        if add_self_loop:
            A = A + torch.eye(A.size(0), device=A.device)

        D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
        A_norm = D @ A @ D
        return A_norm


class GCNLayerNoPool(GCNLayer):
    """
    Stacked GCN Blocks without finger pooling at the end\n
    Input: (num_landmarks=11, num_dimensions=3)\n
    Output: (gcn_features*num_landmarks)
    """

    def forward(self, X):
        out = self.net(X)
        out = out.reshape(out.size(0), -1)
        return out


class GCNBlock(nn.Module):
    """
    Graph Convolution Network with residual, dropout\n
    Input: (B, num_landmarks=11, in_dim)\n
    Output: (B, num_landmarks=11, out_dim)
    """

    def __init__(self, in_dim, out_dim, dropout, A1, A2):
        super().__init__()
        self.register_buffer("A1", A1)
        self.register_buffer("A2", A2)
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)

        self.residual = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        AX1 = torch.matmul(self.A1, X)  # type: ignore
        AX2 = torch.matmul(self.A2, X)  # type: ignore

        res = self.residual(X)
        out = self.W1(AX1) + self.W2(AX2)
        out = out + res
        out = self.bn(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = self.dropout(out)
        return out
