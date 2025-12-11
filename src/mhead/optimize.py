import time
import optuna
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from src import DEVICE
from src.gtcn.dataset import GTCNDataset
from src.mhead import DEFAULT_TRAINSET_PATH
from src.mhead.model import GTCNMHead, GTCNModelParams
from src.mhead.train import train_one_epoch


RANDOM_SEED = 42
NUM_FOLDS = 5
EPOCHS = 10  # per fold


def validate(model, val_loader, gesture_criterion, none_criterion, bce_weight=1.0):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

            gesture_logits, none_logit = model(x)

            gesture_target = y.clone()
            none_target = (y == 0).float()  # 1: none gesture, 0: real gesture

            mask = y != 0  # all real gestures
            if mask.any():
                loss_gesture = gesture_criterion(
                    gesture_logits[mask],
                    gesture_target[mask] - 1,  # shift target to start from 0
                )
            else:
                loss_gesture = torch.tensor(0.0, device=x.device)
            loss_none = none_criterion(none_logit, none_target)

            loss = loss_gesture + bce_weight * loss_none
            val_loss += loss.item()

    return val_loss / len(val_loader)


class Objective:
    def __init__(self, study_name, training_set_path):
        self.study_name = study_name
        self.training_set_path = training_set_path

    def __call__(self, trial):
        model_params = GTCNModelParams(
            id=f"study_{self.study_name}_trial_{trial.number}",
            GCN_HIDDEN_DIM=trial.suggest_categorical("GCN_HIDDEN_DIM", [16, 32]),
            GCN_DROPOUT=trial.suggest_categorical("GCN_DROPOUT", [0.2, 0.3]),
            TCN_HIDDEN_DIM=trial.suggest_categorical("TCN_HIDDEN_DIM", [64, 128]),
            TCN_KERNEL_SIZE=trial.suggest_categorical("TCN_KERNEL_SIZE", [3, 5]),
            TCN_DILATIONS=trial.suggest_categorical(
                "TCN_DILATIONS", [[1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8, 16]]
            ),
            TCN_DROPOUT=trial.suggest_categorical("TCN_DROPOUT", [0.2, 0.3]),
            CLASS_HIDDEN_DIM=trial.suggest_categorical(
                "CLASS_HIDDEN_DIM", [32, 64, 128]
            ),
        )

        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
        bce_weight = trial.suggest_categorical("bce_weight", [0.1, 0.5, 1.0])

        # load dataset
        with open(self.training_set_path, "rb") as f:
            data = pickle.load(f)
        X = data["X"]
        y = data["y"]
        seq_ids = data["seq_ids"]

        # determine sequences in each fold
        unique_seqs = np.unique(seq_ids)
        np.random.seed(RANDOM_SEED)
        shuffled_seqs = np.random.permutation(unique_seqs)
        seq_folds = np.array_split(shuffled_seqs, NUM_FOLDS)

        # perform k-fold cross-validation
        scores = []
        print("=" * 50)
        for fold_idx, val_seqs in enumerate(seq_folds):
            print(f"Fold {fold_idx + 1}/{NUM_FOLDS}")
            print(f"> val sequences: {sorted(val_seqs.tolist())}")

            train_idx = np.where(~np.isin(seq_ids, val_seqs))[0]
            val_idx = np.where(np.isin(seq_ids, val_seqs))[0]

            model = GTCNMHead(model_params).to(DEVICE)
            train_loader = DataLoader(
                GTCNDataset(X[train_idx], y[train_idx]), batch_size=32, shuffle=True
            )
            val_loader = DataLoader(
                GTCNDataset(X[val_idx], y[val_idx]), batch_size=32, shuffle=False
            )

            gesture_criterion = torch.nn.CrossEntropyLoss()
            none_criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
            )

            # Train
            for _ in range(EPOCHS):
                train_one_epoch(
                    model,
                    train_loader,
                    gesture_criterion,
                    none_criterion,
                    optimizer,
                    bce_weight,
                )

            # Validate
            fold_loss = validate(
                model, val_loader, gesture_criterion, none_criterion, bce_weight
            )
            scores.append(fold_loss)

            print(f"> loss: {fold_loss:.4f}")

        return sum(scores) / len(scores)


def find_optimize_params(study_name, n_trials, training_set_path):
    start_time = time.time()

    print(f"Starting study '{study_name}' with {n_trials} trials...")
    objective = Objective(study_name, training_set_path)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.RandomSampler()
    )
    study.optimize(objective, n_trials)

    print("=" * 50)
    print(f"Optimization completed in {time.time() - start_time:.2f} seconds.")
    print("Best Trial:", study.best_trial.number)
    print("Best Score:", study.best_trial.value)
    print("Best Params:", study.best_trial.params)

    return study.best_trial.params


if __name__ == "__main__":
    find_optimize_params(
        study_name="default",
        n_trials=10,
        training_set_path=DEFAULT_TRAINSET_PATH,
    )
