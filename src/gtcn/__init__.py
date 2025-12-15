from src.utils import GestureLabel, HandLandmark
import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset


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
OUTPUT_GESTURES = [
    GestureLabel.NONE,
    GestureLabel.GRAB,
    GestureLabel.PINCH,
    GestureLabel.TAP,
    GestureLabel.DENY,
    GestureLabel.KNOB,
    GestureLabel.EXPAND,
]


class GTCNDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, window_length: int):
        self.check_shape(X, y, window_length)
        print(f"> Creating GestureDataset with {y.shape[0]} samples.")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @staticmethod
    def check_shape(X: np.ndarray, y: np.ndarray, window_length: int):
        assert X.shape[0] == y.shape[0], "X and y length mismatch!"
        assert X.shape[1] == window_length, "X window length mismatch!"
        assert X.shape[2] == len(INPUT_LANDMARKS), "X landmark count mismatch!"
        assert X.shape[3] == INPUT_DIMENSIONS, "X coordinate dimension mismatch!"

        y_unique = np.unique(y)
        for label in y_unique:
            assert 0 <= label < len(OUTPUT_GESTURES), "y label value out of range!"

    @staticmethod
    def print_label_distribution(y: np.ndarray):
        counts = Counter(y)
        print("> Label distribution:")
        for gesture in OUTPUT_GESTURES:
            idx = OUTPUT_GESTURES.index(gesture)
            print(f"  {gesture.name}:{counts[idx]};")
