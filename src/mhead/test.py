import time
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from src import DEVICE
from src.gtcn.dataset import GTCNDataset
from src.mhead import DEFAULT_TESTSET_PATH, DEFAULT_MODEL_PATH
from src.mhead.model import GTCNMHead, GTCNModelParams


def load_model(model_path: str) -> GTCNMHead:
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Reconstruct hyperparameters
    hyperparams_dict = checkpoint["hyperparams"]
    hyperparams = GTCNModelParams(
        id=hyperparams_dict.get("id", "model"),
        GCN_HIDDEN_DIM=hyperparams_dict["GCN_HIDDEN_DIM"],
        GCN_DROPOUT=hyperparams_dict["GCN_DROPOUT"],
        TCN_HIDDEN_DIM=hyperparams_dict["TCN_HIDDEN_DIM"],
        TCN_KERNEL_SIZE=hyperparams_dict["TCN_KERNEL_SIZE"],
        TCN_DILATIONS=hyperparams_dict["TCN_DILATIONS"],
        TCN_DROPOUT=hyperparams_dict["TCN_DROPOUT"],
        CLASS_HIDDEN_DIM=hyperparams_dict["CLASS_HIDDEN_DIM"],
    )

    # Load model
    model = GTCNMHead(hyperparams).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model Hyperparameters: {hyperparams}")

    return model


def test_one_dataset(model, test_loader):
    model.eval()
    truths = []
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            x, y_batch = batch
            x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)

            gesture_logits, none_logit = model(x)

            none_prob = torch.sigmoid(none_logit)
            pred_y = torch.where(
                none_prob > 0.5,
                torch.zeros_like(none_logit, dtype=torch.long),  # NONE
                torch.argmax(gesture_logits, dim=1)
                + 1,  # shift by 1 to account for NONE
            )

            truths.extend(y_batch.cpu().numpy().tolist())
            predictions.extend(pred_y.cpu().numpy().tolist())

    return truths, predictions


def _majority_vote_smoothing(preds, window_size=5):
    smoothed = []
    n = len(preds)

    # only get previous window since we couldn't use future info in real-time
    for end in range(n):
        start = max(0, end - window_size + 1)
        window = preds[start : end + 1]
        if len(window) >= 1:
            values, counts = np.unique(window, return_counts=True)
            smoothed.append(values[counts.argmax()])
        else:
            smoothed.append(preds[end])
    return np.array(smoothed)


def test_model(test_set_path, model_path, batch_size=32):
    start_time = time.time()

    model = load_model(model_path)

    # Load full dataset
    with open(test_set_path, "rb") as f:
        data = pickle.load(f)
    X = data["X"]
    y = data["y"]
    seq_ids = data["seq_ids"]

    # we need to evaluate each sequence separately to avoid data leakage
    truths = []
    predictions = []
    smoothed_predictions = []
    for seq_id in np.unique(seq_ids):
        idx = np.where(seq_ids == seq_id)[0]
        X_seq = X[idx]
        y_seq = y[idx]

        test_loader = DataLoader(
            GTCNDataset(X_seq, y_seq), batch_size=batch_size, shuffle=False
        )
        seq_truths, seq_predictions = test_one_dataset(model, test_loader)
        seq_smoothed = _majority_vote_smoothing(seq_predictions)

        truths.extend(seq_truths)
        predictions.extend(seq_predictions)
        smoothed_predictions.extend(seq_smoothed.tolist())

    print("\nClassification Report:")
    print(classification_report(truths, predictions, digits=4))

    print("\nClassification Report with Majority Vote Smoothing:")
    print(classification_report(truths, smoothed_predictions, digits=4))

    print(f"\nTest completed in {time.time() - start_time:.2f} seconds.")

    return model


if __name__ == "__main__":
    test_model(
        test_set_path=DEFAULT_TESTSET_PATH,
        model_path=DEFAULT_MODEL_PATH,
    )
