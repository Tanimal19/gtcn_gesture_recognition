import time
import pickle
import torch
from torch.utils.data import DataLoader
from src import DEVICE
from src.gtcn import DEFAULT_TRAINSET_PATH
from src.pthresh import DEFAULT_MODEL_PATH
from src.gtcn.dataset import GTCNDataset
from src.pthresh.model import GTCNPThresh, GTCNModelParams


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    epoch_loss = 0.0

    for batch in train_loader:
        x, y_batch = batch
        x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)

        gesture_logits, _ = model(x)

        mask = y_batch != 0  # all real gestures
        if not mask.any():
            continue

        loss = criterion(gesture_logits[mask], y_batch[mask] - 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    return avg_loss


def train_model(params, epochs, training_set_path, model_path, batch_size=32):
    start_time = time.time()

    print(f"Training GTCNPThresh model with parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # Load dataset
    with open(training_set_path, "rb") as f:
        data = pickle.load(f)
    X = data["X"]
    y = data["y"]

    # Create DataLoader
    train_loader = DataLoader(GTCNDataset(X, y), batch_size=batch_size, shuffle=True)

    # Initialize model
    model_params = GTCNModelParams(
        id="pthresh_model",
        GCN_HIDDEN_DIM=params["GCN_HIDDEN_DIM"],
        GCN_DROPOUT=params["GCN_DROPOUT"],
        TCN_HIDDEN_DIM=params["TCN_HIDDEN_DIM"],
        TCN_KERNEL_SIZE=params["TCN_KERNEL_SIZE"],
        TCN_DILATIONS=params["TCN_DILATIONS"],
        TCN_DROPOUT=params["TCN_DROPOUT"],
        CLASS_HIDDEN_DIM=params["CLASS_HIDDEN_DIM"],
    )
    model = GTCNPThresh(model_params).to(DEVICE)

    # Setup training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

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

    return model, best_loss


if __name__ == "__main__":
    params = {
        "GCN_HIDDEN_DIM": 32,
        "GCN_DROPOUT": 0.2,
        "TCN_HIDDEN_DIM": 128,
        "TCN_KERNEL_SIZE": 3,
        "TCN_DILATIONS": [1, 2, 4, 8],
        "TCN_DROPOUT": 0.2,
        "CLASS_HIDDEN_DIM": 64,
        "learning_rate": 0.001,
    }
    train_model(
        params,
        epochs=10,
        training_set_path=DEFAULT_TRAINSET_PATH,
        model_path=DEFAULT_MODEL_PATH,
    )
