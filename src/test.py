import time
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.utils import DEVICE
from src.gtcn.model import GTCNModel, GTCNParams
from src.dataset_builder import GTCNDataset


def load_model(model_path: str) -> GTCNModel:
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # Reconstruct hyperparameters
    hyperparams_dict = checkpoint["hyperparams"]
    hyperparams = GTCNParams(
        id=hyperparams_dict["id"],
        GCN_CLASS=hyperparams_dict["GCN_CLASS"],
        TCN_CLASS=hyperparams_dict["TCN_CLASS"],
        CLASSIFIER_CLASS=hyperparams_dict["CLASSIFIER_CLASS"],
        WINDOW_LENGTH=hyperparams_dict["WINDOW_LENGTH"],
        GCN_DIMS=hyperparams_dict["GCN_DIMS"],
        GCN_DROPOUT=hyperparams_dict["GCN_DROPOUT"],
        TCN_CHANNELS=hyperparams_dict["TCN_CHANNELS"],
        TCN_KERNEL_SIZE=hyperparams_dict["TCN_KERNEL_SIZE"],
        TCN_DILATIONS=hyperparams_dict["TCN_DILATIONS"],
        TCN_DROPOUT=hyperparams_dict["TCN_DROPOUT"],
        CLASSIFIER_DIM=hyperparams_dict["CLASSIFIER_DIM"],
        DOUBLE_HEAD_BCE_WEIGHT=hyperparams_dict["DOUBLE_HEAD_BCE_WEIGHT"],
        PROB_THRESHOLD=hyperparams_dict["PROB_THRESHOLD"],
    )

    # Load model
    model = GTCNModel(hyperparams).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"model hyperparameters: {hyperparams}")

    return model


def test_one_sequence(model: GTCNModel, seq_loader: DataLoader):
    model.eval()
    truths = []
    predictions = []

    with torch.no_grad():
        for batch in seq_loader:
            x, y_batch = batch
            x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)

            output = model(x)
            pred_y = model.inference_gesture(output)

            truths.extend(y_batch.cpu().numpy().tolist())
            predictions.extend(pred_y.cpu().numpy().tolist())

    return truths, predictions


def test_model(test_dataset_path, model_path, batch_size=32):
    start_time = time.time()

    print(f"+ Start testing model from {model_path} on dataset {test_dataset_path}")

    model = load_model(model_path)

    # Load full dataset
    with open(test_dataset_path, "rb") as f:
        data = pickle.load(f)
    X = data["X"]
    y = data["y"]
    seq_ids = data["seq_ids"]

    # we need to evaluate each sequence separately to avoid data leakage
    results = []
    for seq_id in np.unique(seq_ids):
        idx = np.where(seq_ids == seq_id)[0]
        X_seq = X[idx]
        y_seq = y[idx]

        seq_loader = DataLoader(
            GTCNDataset(X_seq, y_seq, model.hyperparams.WINDOW_LENGTH),
            batch_size=batch_size,
            shuffle=False,
        )
        seq_truths, seq_predictions = test_one_sequence(model, seq_loader)

        assert len(seq_truths) == len(idx)
        assert len(seq_predictions) == len(idx)

        results.append(
            {
                "seq_id": seq_id,
                "truths": seq_truths,
                "predictions": seq_predictions,
            }
        )

    print(f"\nTest completed in {time.time() - start_time:.2f} seconds.")

    return results
