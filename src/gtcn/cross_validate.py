import time
import optuna
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from src.gtcn.model import GTCNModel, GTCNHyperParams
from collections import Counter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"> Using device: {DEVICE}")

RANDOM_SEED = 42
NUM_FOLDS = 5


class GTCNDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0], "X and y length mismatch!"
        assert X.shape[1] == GTCNModel.WINDOW_LENGTH, "X window length mismatch!"
        assert X.shape[2] == len(GTCNModel.LANDMARKS), "X landmark count mismatch!"
        assert X.shape[3] == 3, "X coordinate dimension mismatch!"

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _compute_balanced_weights(y, num_classes):
    counts = Counter(y)
    total = len(y)
    weights = np.zeros(num_classes, dtype=np.float32)

    for cls in range(num_classes):
        count = counts[cls] if counts[cls] > 0 else 1
        weights[cls] = total / (num_classes * count)

    return torch.tensor(weights, dtype=torch.float32)


def evaluate_model(
    model: GTCNModel, training_params, train_idx, val_idx, data, batch_size=32
):
    X = data["X"]
    y = data["y"]

    train_loader = DataLoader(
        GTCNDataset(X[train_idx], y[train_idx]), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        GTCNDataset(X[val_idx], y[val_idx]), batch_size=batch_size, shuffle=False
    )

    weights = _compute_balanced_weights(y[train_idx], len(GTCNModel.GESTURES)).to(
        DEVICE
    )

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_params["learning_rate"],
        weight_decay=training_params["weight_decay"],
    )

    EPOCHS = 10  # random search 不要太多，後面 final training 才用多 epoch
    model = model.to(DEVICE)

    # ---------- Train ----------
    for _ in range(EPOCHS):
        model.train()
        for batch in train_loader:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ---------- Validate ----------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)
            val_loss += loss.item()

    return val_loss / len(val_loader)


# -----------------------------------------------------------
# Optuna objective function
# -----------------------------------------------------------
def objective(trial):

    # ---- 隨機選 hyperparameters ----
    model_params = GTCNHyperParams(
        id=f"trial_{trial.number}",
        GCN_HIDDEN_DIM=trial.suggest_categorical("GCN_HIDDEN_DIM", [16, 32, 64]),
        GCN_DROPOUT=trial.suggest_float("GCN_DROPOUT", 0.1, 0.4),
        TCN_HIDDEN_DIM=trial.suggest_categorical("TCN_HIDDEN_DIM", [32, 64, 128]),
        TCN_KERNEL_SIZE=trial.suggest_categorical("TCN_KERNEL_SIZE", [3, 5]),
        TCN_DILATIONS=trial.suggest_categorical(
            "TCN_DILATIONS", [(1, 2, 4), (1, 2, 4, 8), (1, 2, 4, 8, 16)]
        ),
        TCN_DROPOUT=trial.suggest_float("TCN_DROPOUT", 0.1, 0.4),
        CLASS_HIDDEN_DIM=trial.suggest_categorical("CLASS_HIDDEN_DIM", [32, 64, 128]),
    )
    training_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3),
    }

    # load dataset
    with open("./src/gtcn/datasets/train.pkl", "rb") as f:
        data = pickle.load(f)

    # determine sequences in each fold
    seq_ids = data["seq_ids"]
    unique_seqs = np.unique(seq_ids)
    np.random.seed(RANDOM_SEED)
    shuffled_seqs = np.random.permutation(unique_seqs)
    seq_folds = np.array_split(shuffled_seqs, NUM_FOLDS)

    # perform k-fold cross-validation
    scores = []
    for fold_idx, val_seqs in enumerate(seq_folds):
        print(f"====================")
        print(f" Fold {fold_idx + 1}/{NUM_FOLDS}")
        print(f" Validation sequences: {sorted(val_seqs.tolist())}")
        print(f"====================")

        train_idx = np.where(~np.isin(seq_ids, val_seqs))[0]
        val_idx = np.where(np.isin(seq_ids, val_seqs))[0]

        model = GTCNModel(model_params).to(DEVICE)
        fold_loss = evaluate_model(model, training_params, train_idx, val_idx, data)
        scores.append(fold_loss)

    return sum(scores) / len(scores)


# -----------------------------------------------------------
# Run Optuna Random Search
# -----------------------------------------------------------
start_time = time.time()
study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.RandomSampler()
)
study.optimize(objective, n_trials=20)

print("Best Trial:", study.best_trial.number)
print("Best Score:", study.best_trial.value)
print("Best Params:", study.best_trial.params)
print("Total Optimization Time: %.2f seconds" % (time.time() - start_time))

# nohup python -u -m src.gtcn.cross_validate &
