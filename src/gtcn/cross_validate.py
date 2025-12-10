import time
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from src.gtcn.model import GTCNModel, GTCNHyperParams
from src.gtcn.create_set import DATASET_FOLDER
from src.gtcn.train import SequenceDataset, _compute_balanced_weights, _train_one_epoch


def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)

            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    report = classification_report(all_labels, all_preds, digits=3, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, report, cm


def cross_validate(learning_rate, epochs, folds, model_params: GTCNHyperParams):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    with open(DATASET_FOLDER + "train.pkl", "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"]
    num_classes = len(GTCNModel.GESTURES)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n====================")
        print(f" Fold {fold + 1}/{folds}")
        print(f"====================")

        # Prepare data
        train_ds = SequenceDataset(X[train_idx], y[train_idx])
        val_ds = SequenceDataset(X[val_idx], y[val_idx])
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        weights = _compute_balanced_weights(y[train_idx], num_classes).to(device)

        model = GTCNModel(model_params).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training
        for epoch in range(epochs):
            train_loss = _train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            print(
                f"[Fold {fold+1}] Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}"
            )

        # Evaluation
        val_loss, report, cm = _evaluate(model, val_loader, criterion, device)

        print(f"\n[Fold {fold+1}] Validation Loss: {val_loss:.4f}")
        print("\nClassification Report:")
        print(report)

        print("Confusion Matrix:")
        print(cm)

        fold_results.append(report["macro avg"]["f1-score"])

    print("\n====================")
    print(" Final Cross-Validation Results")
    print("====================")

    for i, f1 in enumerate(fold_results):
        print(f"Fold {i+1}: Macro F1 = {f1:.4f}")

    print(f"\nAverage Macro F1: {np.mean(fold_results):.4f}")


if __name__ == "__main__":
    start_time = time.time()

    # cross validation to find best hyperparameters
    model_params_list = [
        GTCNHyperParams(
            id="default",
            GCN_HIDDEN_DIM=16,
            GCN_DROPOUT=0.2,
            TCN_HIDDEN_DIM=64,
            TCN_KERNEL_SIZE=3,
            TCN_DILATIONS=[1, 2, 4, 8],
            TCN_DROPOUT=0.2,
            CLASS_HIDDEN_DIM=64,
        ),
    ]

    for params in model_params_list:
        print(f"Training GTCN Model with params ID: {params.id}")
        cross_validate(
            learning_rate=1e-3,
            epochs=10,
            folds=5,
            model_params=params,
        )
        print(f"Completed with time: {time.time() - start_time}\n")
