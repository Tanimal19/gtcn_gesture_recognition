import os
import sys
import pickle
import torch
import time
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
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


def load_all_sequences(max_sequence_id=None):
    seq_folder = DATASET_FOLDER + "training_set/"
    seq_files = sorted(f for f in os.listdir(seq_folder) if f.endswith(".pkl"))
    print(f"Found {len(seq_files)} sequence files.")

    sequences = {}
    for seq_id, seq_file in enumerate(seq_files):
        if max_sequence_id is not None and seq_id > max_sequence_id:
            break

        with open(seq_folder + seq_file, "rb") as f:
            data = pickle.load(f)

        sequences[seq_id] = (data["X"], data["y"])
        print(f"Loaded seq {seq_id}: X={data['X'].shape}, y={data['y'].shape}")
    return sequences


def compute_class_weights(y, num_classes):
    counts = Counter(y)
    weights = np.zeros(num_classes, dtype=np.float32)
    for cls in range(num_classes):
        weights[cls] = 1.0 / (counts[cls] if counts[cls] > 0 else 1.0)

    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def train_one_fold(train_loader, val_loader, model, criterion, optimizer, device):
    model.to(device)

    for epoch in range(20):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, train_loss={total_loss:.4f}")

    # Validation
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            p = logits.argmax(dim=1)
            preds.extend(p.cpu().numpy())
            trues.extend(y.cpu().numpy())

    print("Validation result:")
    print(classification_report(trues, preds, digits=3, zero_division=0))

    return model


def main():
    start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sequences = load_all_sequences(50)
    seq_ids = sorted(sequences.keys())

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold_idx, (train_ids, val_ids) in enumerate(kf.split(seq_ids)):
        print("\n============================")
        print(f"LOSO Fold {fold_idx+1}/10")
        print(f"Train sequences: {train_ids}")
        print(f"Val sequences: {val_ids}")
        print("============================")

        # Build train set
        X_train, y_train, X_val, y_val = [], [], [], []
        for tid in train_ids:
            X_seq, y_seq = sequences[seq_ids[tid]]
            X_train.append(X_seq)
            y_train.append(y_seq)
        for vid in val_ids:
            X_seq, y_seq = sequences[seq_ids[vid]]
            X_val.append(X_seq)
            y_val.append(y_seq)
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        X_val = np.concatenate(X_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)
        c = Counter(y_train)
        distribution_str = "training set label distribution:"
        for label_idx in range(len(GTCNModel.GESTURES)):
            distribution_str += f" {GTCNModel.GESTURES[label_idx].name}:{c[label_idx]}"
        print(distribution_str)
        c = Counter(y_val)
        distribution_str = "validation set label distribution:"
        for label_idx in range(len(GTCNModel.GESTURES)):
            distribution_str += f" {GTCNModel.GESTURES[label_idx].name}:{c[label_idx]}"
        print(distribution_str)

        train_ds = SequenceDataset(X_train, y_train)
        val_ds = SequenceDataset(X_val, y_val)

        class_weights = compute_class_weights(y_train, len(GTCNModel.GESTURES))
        print(f"Class weights: {class_weights.numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

        model = GTCNModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_one_fold(train_loader, val_loader, model, criterion, optimizer, device)

        sys.stdout.flush()

    print(f"All LOSO folds finished! (time: {time.time() - start_time} seconds)")


if __name__ == "__main__":
    main()
