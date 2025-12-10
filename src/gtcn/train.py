import time
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.gtcn.model import GTCNHyperParams, GTCNDataset, GTCNModel
from collections import Counter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"> Using device: {DEVICE}")


def compute_balanced_weights(y, num_classes):
    counts = Counter(y)
    total = len(y)
    weights = np.zeros(num_classes, dtype=np.float32)

    for cls in range(num_classes):
        count = counts[cls] if counts[cls] > 0 else 1
        weights[cls] = total / (num_classes * count)

    return torch.tensor(weights, dtype=torch.float32)


def train_final_model(params, epochs, batch_size=32):
    start_time = time.time()

    print("\n" + "=" * 50)
    print(f"Training final model with best parameters {params}")

    # Load full dataset
    with open("./src/gtcn/datasets/train.pkl", "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"]

    # Create full training dataset
    train_loader = DataLoader(GTCNDataset(X, y), batch_size=batch_size, shuffle=True)

    # Extract model and training params
    model_params = GTCNHyperParams(
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
    weights = compute_balanced_weights(y, len(GTCNModel.GESTURES)).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    # Training loop
    best_loss = float("inf")
    for epoch in range(epochs):
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

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "hyperparams": model_params.__dict__,
                },
                "./src/gtcn/models/best_model.pth",
            )

    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds.")

    return model


if __name__ == "__main__":
    example_params = {
        "GCN_HIDDEN_DIM": 16,
        "GCN_DROPOUT": 0.2,
        "TCN_HIDDEN_DIM": 32,
        "TCN_KERNEL_SIZE": 3,
        "TCN_DILATIONS": (1, 3, 9),
        "TCN_DROPOUT": 0.2,
        "CLASS_HIDDEN_DIM": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
    }
    train_final_model(example_params, epochs=10)
