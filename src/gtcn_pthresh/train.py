import os
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from src.gtcn.model import GTCNModel
from src.gtcn.preprocess import DATASET_FOLDER


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        assert X.shape[0] == y.shape[0], "X and y length mismatch!"
        assert X.shape[1] == GTCNModel.WINDOW_LENGTH, "X window length mismatch!"
        assert X.shape[2] == len(GTCNModel.LANDMARKS), "X landmark count mismatch!"
        assert X.shape[3] == 3, "X coordinate dimension mismatch!"

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_one_sequence(model, loader, criterion, optimizer, device):
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


def validate_one_sequence(model, loader, criterion, device):
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


def train_over_sequences(
    model,
    criterion,
    optimizer,
    batch_size,
    epochs,
    device,
):
    seq_folder = DATASET_FOLDER + "training_set/"

    model.to(device)

    seq_files = sorted(f for f in os.listdir(seq_folder) if f.endswith(".pkl"))
    print(f"Found {len(seq_files)} sequence files.")

    for epoch in range(epochs):
        print(f"\n==================== Epoch {epoch+1}/{epochs} ====================")

        total_train_loss = 0
        total_val_loss = 0
        total_val_acc = 0
        total_val_count = 0
        total_samples = 0
        best_train_loss = float("inf")

        # Loop over each sequence file
        for seq_file in seq_files:
            path = os.path.join(seq_folder, seq_file)

            with open(path, "rb") as f:
                data = pickle.load(f)

            X_train, y_train = data["X_train"], data["y_train"]
            X_val, y_val = data["X_val"], data["y_val"]
            total_samples += len(y_train) + len(y_val)

            # Create PyTorch Dataset & DataLoader
            train_ds = SequenceDataset(X_train, y_train)
            val_ds = SequenceDataset(X_val, y_val)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            # Train on this sequence
            seq_train_loss = train_one_sequence(
                model, train_loader, criterion, optimizer, device
            )
            total_train_loss += seq_train_loss * len(train_ds)

            # Validate on this sequence
            seq_val_loss, seq_val_acc = validate_one_sequence(
                model, val_loader, criterion, device
            )

            total_val_loss += seq_val_loss * len(val_ds)
            total_val_acc += seq_val_acc * len(val_ds)
            total_val_count += len(val_ds)

            print(
                f"[{seq_file}] Train Loss={seq_train_loss:.4f} | "
                f"Val Loss={seq_val_loss:.4f} Acc={seq_val_acc:.3f}"
            )

        # Epoch summary
        avg_train_loss = total_train_loss / total_samples
        avg_val_loss = total_val_loss / total_val_count
        avg_val_acc = total_val_acc / total_val_count

        print(
            f"\nEPOCH {epoch+1} SUMMARY:"
            f"\nTrain Loss = {avg_train_loss:.4f}"
            f"\nVal Loss   = {avg_val_loss:.4f}"
            f"\nVal Acc    = {avg_val_acc:.3f}\n"
        )

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), DATASET_FOLDER + "mat/" + "best_model.pth")
            print("Saved new best model.")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GTCNModel()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_over_sequences(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=32,
        epochs=2,
        device=device,
    )
