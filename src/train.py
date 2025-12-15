import time
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter
from src.utils import DEVICE, GestureLabel
from src.gtcn import OUTPUT_GESTURES, GTCNDataset
from src.gtcn.model import GTCNModel, GTCNParams
from src.gtcn.gcn import GCNLayerFingerPool
from src.gtcn.tcn import TCNLayerLastStep
from src.gtcn.classifier import RegularClassifier
from src.dataset_builder import GTCNDataset, DEFAULT_DATASET_FOLDER
from dataclasses import dataclass


DEFAULT_MODEL_FOLDER = "./src/models/"


@dataclass
class GTCNTrainParams:
    model_params: GTCNParams
    training_dataset_path: str
    model_path: str
    weight_mode: str | None = None  # 'simple', 'balanced', or None
    batch_size: int = 32
    learning_rate: float = 2e-3
    epochs: int = 50


def compute_weights(y, mode: str | None = None) -> torch.Tensor:
    """mode: 'simple', 'balanced', None"""

    weights = np.zeros(len(OUTPUT_GESTURES), dtype=np.float32)

    if mode == "simple":
        for gesture in OUTPUT_GESTURES:
            idx = OUTPUT_GESTURES.index(gesture)
            weights[idx] = 0.1 if gesture == GestureLabel.NONE else 1.0

    elif mode == "balanced":
        counts = Counter(y)
        total = len(y)
        for gesture in OUTPUT_GESTURES:
            idx = OUTPUT_GESTURES.index(gesture)
            weights[idx] = total / (len(OUTPUT_GESTURES) * counts.get(idx, 1))

    else:
        # no weighting
        weights.fill(1.0)

    print(f"> Label weights: {weights}")

    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: GTCNModel,
    train_loader: DataLoader,
    criterions: list[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
):
    model.train()
    epoch_loss = 0.0

    for batch in train_loader:
        x, y_batch = batch
        x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)

        output = model(x)
        loss = model.compute_loss(output, y_batch, criterions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    return avg_loss


def train_model(params: GTCNTrainParams):
    start_time = time.time()

    print(f"+ Start training model with parameters {params}")

    # Load full dataset
    with open(params.training_dataset_path, "rb") as f:
        data = pickle.load(f)
    X = data["X"]
    y = data["y"]

    train_loader = DataLoader(
        GTCNDataset(X, y, params.model_params.WINDOW_LENGTH),
        batch_size=params.batch_size,
        shuffle=True,
    )

    # Extract model and training params
    model = GTCNModel(params.model_params).to(DEVICE)

    # Setup training
    weights = compute_weights(y, params.weight_mode).to(DEVICE)
    criterions = model.classifier.init_criterions(weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.learning_rate,
    )

    # Training loop
    loss_history: list[float] = []
    best_loss = float("inf")
    early_stop_counter = 0
    early_stop_patience = 10

    for epoch in range(params.epochs):
        epoch_loss = train_one_epoch(model, train_loader, criterions, optimizer)
        print(f"Epoch [{epoch+1}/{params.epochs}], Loss: {epoch_loss:.4f}")
        loss_history.append(epoch_loss)

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "hyperparams": params.model_params.__dict__,
                },
                params.model_path,
            )
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds.")

    return loss_history
