import time
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter
from src import DEVICE
from src.gtcn import DEFAULT_TRAINSET_PATH, DEFAULT_MODEL_PATH
from src.gtcn.model import GTCNModel, GTCNModelParams
from src.gtcn.dataset import GTCNDataset
from src.dataset_utils import GestureLabel


def compute_weights(y, mode: str = "") -> torch.Tensor:
    """mode: 'simple', 'balanced', empty"""

    weights = np.zeros(len(GTCNModel.GESTURES), dtype=np.float32)

    if mode == "simple":
        for gesture in GTCNModel.GESTURES:
            idx = GTCNModel.GESTURES.index(gesture)
            weights[idx] = 0.1 if gesture == GestureLabel.NONE else 1.0

    elif mode == "balanced":
        counts = Counter(y)
        total = len(y)
        for gesture in GTCNModel.GESTURES:
            idx = GTCNModel.GESTURES.index(gesture)
            weights[idx] = total / (len(GTCNModel.GESTURES) * counts.get(idx, 1))

    else:
        # no weighting
        weights.fill(1.0)

    print(f"> Label weights: {weights}")

    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    epoch_loss = 0.0

    for batch in train_loader:
        x, y_batch = batch
        x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    return avg_loss


def train_model(params, epochs, training_set_path, model_path, batch_size=32):
    start_time = time.time()

    print(f"Training model with parameters {params}")

    # Load full dataset
    with open(training_set_path, "rb") as f:
        data = pickle.load(f)
    X = data["X"]
    y = data["y"]

    # Create full training dataset
    train_loader = DataLoader(GTCNDataset(X, y), batch_size=batch_size, shuffle=True)

    # Extract model and training params
    model_params = GTCNModelParams(
        id="best_model",
        GCN_HIDDEN_DIM=params["GCN_HIDDEN_DIM"],
        GCN_DROPOUT=params["GCN_DROPOUT"],
        TCN_HIDDEN_DIM=params["TCN_HIDDEN_DIM"],
        TCN_KERNEL_SIZE=params["TCN_KERNEL_SIZE"],
        TCN_DILATIONS=params["TCN_DILATIONS"],
        TCN_DROPOUT=params["TCN_DROPOUT"],
        CLASS_HIDDEN_DIM=params["CLASS_HIDDEN_DIM"],
    )
    model = GTCNModel(model_params).to(DEVICE)

    # Setup training
    weights = compute_weights(y, "balanced").to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
    )

    # Training loop
    best_loss = float("inf")
    early_stop_counter = 0
    early_stop_patience = 10

    for epoch in range(epochs):
        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "hyperparams": model_params.__dict__,
                },
                model_path,
            )
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds.")

    return model


if __name__ == "__main__":
    print(f"> Using device: {DEVICE}")
    example_params = {
        "GCN_HIDDEN_DIM": 16,
        "GCN_DROPOUT": 0.3,
        "TCN_HIDDEN_DIM": 64,
        "TCN_KERNEL_SIZE": 5,
        "TCN_DILATIONS": (1, 2, 4, 8, 16),
        "TCN_DROPOUT": 0.3,
        "CLASS_HIDDEN_DIM": 32,
        "learning_rate": 1e-3,
    }
    train_model(
        example_params,
        training_set_path=DEFAULT_TRAINSET_PATH,
        model_path=DEFAULT_MODEL_PATH,
        epochs=10,
    )
