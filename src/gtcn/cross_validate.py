import time
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
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

    report = classification_report(
        all_labels, all_preds, digits=3, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, report, cm


def cross_validate(learning_rate, epochs, model_params: GTCNHyperParams, num_folds=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    with open(DATASET_FOLDER + "train.pkl", "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"]
    seq_ids = data["seq_ids"]
    num_classes = len(GTCNModel.GESTURES)

    # Get unique sequences and group them into folds
    unique_seqs = np.unique(seq_ids)
    num_sequences = len(unique_seqs)

    # Shuffle sequences for better distribution across folds
    np.random.seed(42)
    shuffled_seqs = np.random.permutation(unique_seqs)

    # Split sequences into folds
    seq_folds = np.array_split(shuffled_seqs, num_folds)

    print(f"Performing {num_folds}-Fold Grouped Cross-Validation")
    print(f"Total sequences: {num_sequences}")
    print(f"Sequences per fold: {[len(fold) for fold in seq_folds]}")

    fold_results = []

    for fold_idx, test_seqs in enumerate(seq_folds):
        print(f"\n====================")
        print(f" Fold {fold_idx + 1}/{num_folds}")
        print(f" Validation sequences: {sorted(test_seqs.tolist())}")
        print(f"====================")

        # Split by sequence groups: all windows from test_seqs go to validation
        train_mask = ~np.isin(seq_ids, test_seqs)
        val_mask = np.isin(seq_ids, test_seqs)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

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
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"[Fold {fold_idx+1}] Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}"
                )

        # Evaluation
        val_loss, report, cm = _evaluate(model, val_loader, criterion, device)

        print(f"\n[Fold {fold_idx+1}] Validation Loss: {val_loss:.4f}")
        print(f"[Fold {fold_idx+1}] Macro F1: {report['macro avg']['f1-score']:.4f}")
        print(f"[Fold {fold_idx+1}] Accuracy: {report['accuracy']:.4f}")

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "test_seqs": sorted(test_seqs.tolist()),
                "f1_score": report["macro avg"]["f1-score"],
                "accuracy": report["accuracy"],
                "report": report,
                "cm": cm,
            }
        )

    print("\n====================")
    print(f" {num_folds}-Fold Grouped Cross-Validation Results")
    print("====================")

    f1_scores = [r["f1_score"] for r in fold_results]
    accuracies = [r["accuracy"] for r in fold_results]

    for result in fold_results:
        seqs_str = ",".join(map(str, result["test_seqs"]))
        print(
            f"Fold {result['fold']} (seqs {seqs_str}): F1={result['f1_score']:.4f}, Acc={result['accuracy']:.4f}"
        )

    print(f"\nAverage Macro F1: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
    print(f"Average Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")

    return fold_results


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
        results = cross_validate(
            learning_rate=1e-3,
            epochs=10,
            model_params=params,
            num_folds=5,
        )
        print(f"Completed with time: {time.time() - start_time:.2f}s\n")


# nohup python -u -m src.gtcn.cross_validate