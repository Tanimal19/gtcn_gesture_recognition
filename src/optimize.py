import time
import optuna
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.utils import DEVICE
from src.dataset_builder import GTCNDataset
from src.train import compute_weights, train_one_epoch, GTCNTrainParams
from src.gtcn.model import GTCNModel, GTCNParams
from src.gtcn.classifier import (
    RegularClassifier,
    DoubleHeadClassifier,
    ProbThresholdClassifier,
)
import copy


RANDOM_SEED = 42
NUM_FOLDS = 3
EPOCHS = 10


def validate(
    model: GTCNModel, val_loader: DataLoader, criterions: list[torch.nn.Module]
) -> float:
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            x, y_batch = batch
            x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)

            output = model(x)
            loss = model.compute_loss(output, y_batch, criterions)
            val_loss += loss.item()

    return val_loss / len(val_loader)


class Objective:
    def __init__(self, study_name, train_dataset_path, base_params: GTCNParams):
        self.study_name = study_name
        self.train_dataset_path = train_dataset_path
        self.base_params = base_params

    def __call__(self, trial: optuna.trial.Trial):

        model_params = copy.deepcopy(self.base_params)

        # gcn params
        model_params.GCN_DIMS = trial.suggest_categorical(
            "GCN_DIMS", [[16], [16, 32], [16, 32, 64]]  # type: ignore
        )
        model_params.GCN_DROPOUT = trial.suggest_categorical("GCN_DROPOUT", [0.2, 0.3])

        # tcn params
        tcn_params = [
            ([64, 64, 64], 3, [1, 2, 4]),  # default
            ([128, 128], 5, [1, 2]),  # less layers, larger kernel
            ([64, 128, 256], 3, [1, 2, 4]),  # increasing channels
            ([64, 64, 64], 3, [1, 3, 9]),  # larger dilation
        ]
        tcn_choice = trial.suggest_categorical("TCN_PARAMS", tcn_params)  # type: ignore
        model_params.TCN_CHANNELS = tcn_choice[0]
        model_params.TCN_KERNEL_SIZE = tcn_choice[1]
        model_params.TCN_DILATIONS = tcn_choice[2]
        model_params.TCN_DROPOUT = trial.suggest_categorical("TCN_DROPOUT", [0.2, 0.3])

        # classifier params
        model_params.CLASSIFIER_DIM = trial.suggest_categorical(
            "CLASSIFIER_DIM", [32, 64, 128]
        )
        if model_params.CLASSIFIER_CLASS.__name__ == DoubleHeadClassifier.__name__:
            model_params.DOUBLE_HEAD_BCE_WEIGHT = trial.suggest_categorical(
                "DOUBLE_HEAD_BCE_WEIGHT", [0.1, 0.5, 1.0]
            )
        elif model_params.CLASSIFIER_CLASS.__name__ == ProbThresholdClassifier.__name__:
            model_params.PROB_THRESHOLD = trial.suggest_categorical(
                "PROB_THRESHOLD", [0.5, 0.6, 0.7, 0.8, 0.9]
            )

        # training params
        learning_rate = trial.suggest_categorical("learning_rate", [2e-3, 1e-2, 5e-4])

        # load dataset
        with open(self.train_dataset_path, "rb") as f:
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

            model = GTCNModel(model_params).to(DEVICE)

            train_idx = np.where(~np.isin(seq_ids, val_seqs))[0]
            val_idx = np.where(np.isin(seq_ids, val_seqs))[0]
            train_loader = DataLoader(
                GTCNDataset(X[train_idx], y[train_idx], model_params.WINDOW_LENGTH),
                batch_size=32,
                shuffle=True,
            )
            val_loader = DataLoader(
                GTCNDataset(X[val_idx], y[val_idx], model_params.WINDOW_LENGTH),
                batch_size=32,
                shuffle=False,
            )

            weights = compute_weights(y[train_idx], "simple").to(DEVICE)
            criterions = model.classifier.init_criterions(weights)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
            )

            # Train
            for _ in range(EPOCHS):
                train_one_epoch(model, train_loader, criterions, optimizer)

            # Validate
            fold_loss = validate(model, val_loader, criterions)
            scores.append(fold_loss)

            print(f"> loss: {fold_loss:.4f}")

        return sum(scores) / len(scores)


def run_optimize(
    study_name, n_trials, training_set_path, base_params: GTCNParams
) -> GTCNTrainParams:
    start_time = time.time()

    print(f"Starting study '{study_name}' with {n_trials} trials...")
    objective = Objective(study_name, training_set_path, base_params)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.RandomSampler()
    )
    study.optimize(objective, n_trials)

    print("=" * 50)
    print(f"Optimization completed in {time.time() - start_time:.2f} seconds.")
    print("Best Trial:", study.best_trial.number)
    print("Best Score:", study.best_trial.value)
    print("Best Params:", study.best_trial.params)

    best_model_params = copy.deepcopy(base_params)
    best_model_params.GCN_DIMS = study.best_trial.params["GCN_DIMS"]
    best_model_params.GCN_DROPOUT = study.best_trial.params["GCN_DROPOUT"]
    best_model_params.TCN_CHANNELS = study.best_trial.params["TCN_PARAMS"][0]
    best_model_params.TCN_KERNEL_SIZE = study.best_trial.params["TCN_PARAMS"][1]
    best_model_params.TCN_DILATIONS = study.best_trial.params["TCN_PARAMS"][2]
    best_model_params.TCN_DROPOUT = study.best_trial.params["TCN_DROPOUT"]
    best_model_params.CLASSIFIER_DIM = study.best_trial.params["CLASSIFIER_DIM"]
    if best_model_params.CLASSIFIER_CLASS.__name__ == DoubleHeadClassifier.__name__:
        best_model_params.DOUBLE_HEAD_BCE_WEIGHT = study.best_trial.params[
            "DOUBLE_HEAD_BCE_WEIGHT"
        ]
    elif (
        best_model_params.CLASSIFIER_CLASS.__name__ == ProbThresholdClassifier.__name__
    ):
        best_model_params.PROB_THRESHOLD = study.best_trial.params["PROB_THRESHOLD"]

    return GTCNTrainParams(
        model_params=best_model_params,
        training_dataset_path=training_set_path,
        weight_mode="simple",
        learning_rate=study.best_trial.params["learning_rate"],
    )
