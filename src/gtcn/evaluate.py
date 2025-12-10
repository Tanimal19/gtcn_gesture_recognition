import time
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.gtcn.model import GTCNHyperParams, GTCNModel
from src.gtcn.create_training_set import TrainingDataset
from sklearn.metrics import classification_report

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path: str) -> GTCNModel:
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Reconstruct hyperparameters
    hyperparams_dict = checkpoint["hyperparams"]
    hyperparams = GTCNHyperParams(
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
    model = GTCNModel(hyperparams).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model Hyperparameters: {hyperparams}")

    return model


def majority_vote_smoothing(preds, window_size):
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


def evaluate_model(
    test_set_path,
    model_path,
    batch_size=32,
):
    start_time = time.time()

    model = load_model(model_path)

    # Load full dataset
    with open(test_set_path, "rb") as f:
        data = pickle.load(f)
    X = data["X"]
    y = data["y"]
    seq_ids = data["seq_ids"]

    truths = []
    predictions = []
    smoothed_predictions = []
    for seq_id in np.unique(seq_ids):
        idx = np.where(seq_ids == seq_id)[0]
        X_seq = X[idx]
        y_seq = y[idx]

        val_loader = DataLoader(
            TrainingDataset(X_seq, y_seq), batch_size=batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch in val_loader:
                x, truth_y = batch
                x, truth_y = x.to(DEVICE), truth_y.to(DEVICE)

                logits = model(x)
                pred_y = torch.argmax(logits, dim=1)
                truths.extend(truth_y.cpu().numpy().tolist())
                predictions.extend(pred_y.cpu().numpy().tolist())
                smoothed_predictions.extend(
                    majority_vote_smoothing(
                        pred_y.cpu().numpy().tolist(), window_size=5
                    )
                )

    print("\nClassification Report:")
    print(classification_report(truths, predictions, digits=4))

    print("\nClassification Report after Majority Vote Smoothing:")
    print(classification_report(truths, smoothed_predictions, digits=4))

    print(f"\nEvaluation completed in {time.time() - start_time:.2f} seconds.")

    return model


if __name__ == "__main__":
    print(f"> Using device: {DEVICE}")
    evaluate_model(
        test_set_path="./src/gtcn/datasets/test_s5.pkl",
        model_path="./src/gtcn/datasets/model_s5.pth",
    )
