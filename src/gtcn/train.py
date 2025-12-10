import time
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from src.gtcn.model import GTCNModel, GTCNHyperParams
from src.gtcn.create_set import DATASET_FOLDER
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


def _compute_balanced_weights(y, num_classes):
    counts = Counter(y)
    total = len(y)
    weights = np.zeros(num_classes, dtype=np.float32)

    for cls in range(num_classes):
        count = counts[cls] if counts[cls] > 0 else 1
        weights[cls] = total / (num_classes * count)

    return torch.tensor(weights, dtype=torch.float32)


def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


def train(learning_rate, epochs, model_params: GTCNHyperParams):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(DATASET_FOLDER + "train.pkl", "rb") as f:
        data = pickle.load(f)
    X = data["X"]
    y = data["y"]
    num_classes = len(GTCNModel.GESTURES)

    class_weights = _compute_balanced_weights(y, num_classes).to(device)
    train_ds = SequenceDataset(X, y)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = GTCNModel(model_params).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), DATASET_FOLDER + f"gtcn_{model_params.id}.pth")


if __name__ == "__main__":
    start_time = time.time()

    model_params = GTCNHyperParams(
        id="default",
        GCN_HIDDEN_DIM=16,
        GCN_DROPOUT=0.2,
        TCN_HIDDEN_DIM=64,
        TCN_KERNEL_SIZE=3,
        TCN_DILATIONS=[1, 2, 4, 8],
        TCN_DROPOUT=0.2,
        CLASS_HIDDEN_DIM=64,
    )

    train(
        learning_rate=1e-3,
        epochs=10,
        model_params=model_params,
    )
    print(f"Training completed in {time.time() - start_time} seconds.")
