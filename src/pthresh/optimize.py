import time
import optuna
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from src import DEVICE
from src.gtcn import DEFAULT_TRAINSET_PATH
from src.gtcn.dataset import GTCNDataset
from src.pthresh.model import GTCNPThresh, GTCNModelParams
from src.pthresh.train import train_one_epoch


RANDOM_SEED = 42
NUM_FOLDS = 5
EPOCHS = 10  # per fold


def validate_with_threshold(model, val_loader, threshold):
    """
    Validate model using probability threshold for NONE prediction.

    Returns:
        f1_macro: Macro F1 score across all gestures
        f1_weighted: Weighted F1 score
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

            _, gesture_probs = model(x)  # (B, num_real_gestures)
            max_probs, max_indices = torch.max(gesture_probs, dim=1)

            # Apply threshold: if max_prob < threshold, predict NONE (0)
            # Otherwise predict the gesture with highest prob (shift by +1)
            preds = torch.where(
                max_probs < threshold, torch.zeros_like(max_indices), max_indices + 1
            )

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return f1_macro, f1_weighted


def optimize_threshold(model, val_loader, threshold_range=None):
    """
    Find optimal probability threshold for NONE detection.

    Args:
        model: Trained GTCNPThresh model
        val_loader: Validation DataLoader
        threshold_range: List of threshold values to test

    Returns:
        best_threshold: Optimal threshold value
        best_f1: Best F1 score achieved
        results: Dictionary with all threshold results
    """
    if threshold_range is None:
        threshold_range = np.arange(0.1, 1.0, 0.05)

    print("\nOptimizing probability threshold...")
    print(f"Testing {len(threshold_range)} threshold values")

    results = {"thresholds": [], "f1_macro": [], "f1_weighted": []}
    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in threshold_range:
        f1_macro, f1_weighted = validate_with_threshold(model, val_loader, threshold)

        results["thresholds"].append(threshold)
        results["f1_macro"].append(f1_macro)
        results["f1_weighted"].append(f1_weighted)

        print(
            f"  Threshold: {threshold:.2f} → F1 Macro: {f1_macro:.4f}, F1 Weighted: {f1_weighted:.4f}"
        )

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_threshold = threshold

    print(f"\n✓ Best threshold: {best_threshold:.2f} (F1 Macro: {best_f1:.4f})")

    return best_threshold, best_f1, results


def evaluate_with_confusion_matrix(model, val_loader, threshold):
    """
    Evaluate model and print confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

            _, gesture_probs = model(x)
            max_probs, max_indices = torch.max(gesture_probs, dim=1)

            preds = torch.where(
                max_probs < threshold, torch.zeros_like(max_indices), max_indices + 1
            )

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)

    print("\nConfusion Matrix:")
    print(cm)

    # Print per-class metrics
    from sklearn.metrics import classification_report
    from src.pthresh.model import GTCNPThresh

    target_names = [g.name for g in GTCNPThresh.GESTURES]
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels, all_preds, target_names=target_names, zero_division=0
        )
    )


class Objective:
    """Optuna objective for hyperparameter optimization."""

    def __init__(self, study_name, training_set_path):
        self.study_name = study_name
        self.training_set_path = training_set_path

    def __call__(self, trial):
        model_params = GTCNModelParams(
            id=f"study_{self.study_name}_trial_{trial.number}",
            GCN_HIDDEN_DIM=trial.suggest_categorical("GCN_HIDDEN_DIM", [16, 32, 64]),
            GCN_DROPOUT=trial.suggest_categorical("GCN_DROPOUT", [0.1, 0.2, 0.3]),
            TCN_HIDDEN_DIM=trial.suggest_categorical("TCN_HIDDEN_DIM", [64, 128, 256]),
            TCN_KERNEL_SIZE=trial.suggest_categorical("TCN_KERNEL_SIZE", [3, 5]),
            TCN_DILATIONS=trial.suggest_categorical(
                "TCN_DILATIONS", [[1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8, 16]]
            ),
            TCN_DROPOUT=trial.suggest_categorical("TCN_DROPOUT", [0.1, 0.2, 0.3]),
            CLASS_HIDDEN_DIM=trial.suggest_categorical(
                "CLASS_HIDDEN_DIM", [32, 64, 128]
            ),
        )

        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        # Load dataset
        with open(self.training_set_path, "rb") as f:
            data = pickle.load(f)
        X = data["X"]
        y = data["y"]
        seq_ids = data["seq_ids"]

        # Determine sequences in each fold
        unique_seqs = np.unique(seq_ids)
        np.random.seed(RANDOM_SEED)
        shuffled_seqs = np.random.permutation(unique_seqs)
        seq_folds = np.array_split(shuffled_seqs, NUM_FOLDS)

        # K-fold cross-validation
        f1_scores = []
        print("=" * 50)
        for fold_idx, val_seqs in enumerate(seq_folds):
            print(f"Fold {fold_idx + 1}/{NUM_FOLDS}")

            train_idx = np.where(~np.isin(seq_ids, val_seqs))[0]
            val_idx = np.where(np.isin(seq_ids, val_seqs))[0]

            model = GTCNPThresh(model_params).to(DEVICE)
            train_loader = DataLoader(
                GTCNDataset(X[train_idx], y[train_idx]), batch_size=32, shuffle=True
            )
            val_loader = DataLoader(
                GTCNDataset(X[val_idx], y[val_idx]), batch_size=32, shuffle=False
            )

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Train for EPOCHS
            for epoch in range(EPOCHS):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
                print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}")

            # Optimize threshold on validation set
            best_threshold, best_f1, _ = optimize_threshold(
                model, val_loader, threshold_range=np.arange(0.1, 0.9, 0.1)
            )
            f1_scores.append(best_f1)

            print(
                f"  → Fold {fold_idx + 1} F1: {best_f1:.4f} (threshold: {best_threshold:.2f})"
            )

        if len(f1_scores) == 0:
            print("\nNo valid folds, returning 0.0")
            return 0.0

        avg_f1 = np.mean(f1_scores)
        print(f"\nAverage F1 across {NUM_FOLDS} folds: {avg_f1:.4f}")
        print("=" * 50)

        return float(avg_f1)


def run_optimize(study_name, n_trials, training_set_path):
    start_time = time.time()

    print(f"Starting study '{study_name}' with {n_trials} trials...")
    objective = Objective(study_name, training_set_path)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(),
    )
    study.optimize(objective, n_trials)

    print("=" * 50)
    print(f"Optimization completed in {time.time() - start_time:.2f} seconds.")
    print(f"Best Trial:", study.best_trial.number)
    print(f"Best Score (F1 Score): {study.best_trial.value:.4f}")
    print(f"Best Params: {study.best_trial.params}")

    return study.best_trial.params


if __name__ == "__main__":
    run_optimize(
        study_name="pthresh_optimization",
        n_trials=1,
        training_set_path=DEFAULT_TRAINSET_PATH,
    )
