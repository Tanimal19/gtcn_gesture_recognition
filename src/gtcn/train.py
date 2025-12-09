import os
import sys
import pickle
import torch
import time
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from src.gtcn.model import GTCNModel
from gtcn.create_set import DATASET_FOLDER
from collections import Counter


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0], "X and y length mismatch!"
        assert X.shape[1] == GTCNModel.WINDOW_LENGTH, "X window length mismatch!"
        assert X.shape[2] == len(GTCNModel.LANDMARKS), "X landmark count mismatch!"
        assert X.shape[3] == 3, "X coordinate dimension mismatch!"

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def stratified_split(
    X: np.ndarray, y: np.ndarray, val_size=0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N = len(y)
    indices = list(range(N))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        stratify=y,
        random_state=42,
    )

    print("> y label distribution:")
    print(f"\ttrain: {Counter(y[train_idx])}")
    print(f"\tval: {Counter(y[val_idx])}")

    return (X[train_idx], y[train_idx], X[val_idx], y[val_idx])


def compute_class_weights(y, num_classes):
    counts = Counter(y)
    weights = np.zeros(num_classes, dtype=np.float32)
    for cls in range(num_classes):
        weights[cls] = 1.0 / (counts[cls] if counts[cls] > 0 else 1.0)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)

    return avg_loss, acc


def train_epochs(learning_rate=1e-3, epochs=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GTCNModel().to(device)

    # loading training data
    with open(DATASET_FOLDER + "train.pkl", "rb") as f:
        data = pickle.load(f)
    class_weights = compute_class_weights(
        data["y"], num_classes=len(GTCNModel.GESTURES)
    )
    X_train, y_train, X_val, y_val = stratified_split(
        data["X"], data["y"], val_size=0.2
    )
    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), DATASET_FOLDER + "best_model.pth")
            print("  > Saved best model.")


if __name__ == "__main__":
    train_epochs(learning_rate=1e-3, epochs=20)
