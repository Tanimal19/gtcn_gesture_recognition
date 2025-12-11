import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from src import DEVICE
from src.gtcn.dataset import GTCNDataset
from src.pthresh.model import GTCNPThresh, GTCNModelParams


def load_model(model_path: str) -> GTCNPThresh:
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
    model = GTCNPThresh(hyperparams).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model Hyperparameters: {hyperparams}")

    return model


def predict_with_threshold(model, x, threshold):
    """
    Make predictions using probability threshold.

    Args:
        model: GTCNPThresh model
        x: Input tensor (B, T, N, C)
        threshold: Probability threshold for NONE detection

    Returns:
        predictions: Predicted class indices (0 for NONE, 1-6 for real gestures)
        max_probs: Maximum probabilities for each sample
    """
    model.eval()
    with torch.no_grad():
        _, gesture_probs = model(x)  # (B, num_real_gestures)
        max_probs, max_indices = torch.max(gesture_probs, dim=1)

        # If max_prob < threshold, predict NONE (0)
        # Otherwise predict the gesture with highest prob (shift by +1)
        preds = torch.where(
            max_probs < threshold, torch.zeros_like(max_indices), max_indices + 1
        )

    return preds, max_probs


def test_model(model_path, test_set_path, threshold=0.5, batch_size=32):
    """
    Evaluate trained model on test set.

    Args:
        model_path: Path to saved model weights
        test_set_path: Path to test dataset pickle file
        threshold: Probability threshold for NONE detection
        batch_size: Batch size for evaluation
    """
    print(f"Loading model from: {model_path}")
    print(f"Using threshold: {threshold:.2f}")

    # Load test data
    with open(test_set_path, "rb") as f:
        data = pickle.load(f)
    X = data["X"]
    y = data["y"]

    test_loader = DataLoader(GTCNDataset(X, y), batch_size=batch_size, shuffle=False)

    model = load_model(model_path)
    model.eval()

    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []

    print("\nEvaluating...")
    with torch.no_grad():
        for batch in test_loader:
            x, y_batch = batch
            x = x.to(DEVICE)

            preds, max_probs = predict_with_threshold(model, x, threshold)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(max_probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    accuracy = np.mean(all_preds == all_labels)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    from src.pthresh.model import GTCNPThresh as Model

    target_names = [g.name for g in Model.GESTURES]
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels, all_preds, target_names=target_names, zero_division=0
        )
    )

    # Probability distribution analysis
    print("\nProbability Distribution:")
    print(f"  Mean max probability: {np.mean(all_probs):.4f}")
    print(f"  Median max probability: {np.median(all_probs):.4f}")
    print(f"  Std max probability: {np.std(all_probs):.4f}")

    # Analyze predictions by true label
    print("\nPredictions by True Label:")
    for label_idx, gesture in enumerate(Model.GESTURES):
        mask = all_labels == label_idx
        if mask.sum() > 0:
            label_probs = all_probs[mask]
            label_preds = all_preds[mask]
            correct = (label_preds == label_idx).sum()
            total = mask.sum()
            print(
                f"  {gesture.name:10s}: Acc={correct}/{total} ({correct/total:.2%}), "
                f"Mean Prob={np.mean(label_probs):.4f}"
            )

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


if __name__ == "__main__":
    from src.pthresh import DEFAULT_MODEL_PATH
    from src.gtcn import DEFAULT_TESTSET_PATH

    results = test_model(
        model_path="./src/pthresh/models/best_model.pth",
        test_set_path=DEFAULT_TESTSET_PATH,
        threshold=0.6,
        batch_size=32,
    )
