import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def calculate_confusion_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate cumulative confusion matrix for a model across all sequences.

    Args:
        df: DataFrame filtered for a specific model

    Returns:
        Confusion matrix as numpy array
    """
    all_truths = []
    all_predictions = []

    for _, row in df.iterrows():
        truths = ast.literal_eval(row["truths"])
        predictions = ast.literal_eval(row["predictions"])
        all_truths.extend(truths)
        all_predictions.extend(predictions)

    # Get all unique labels
    labels = sorted(set(all_truths) | set(all_predictions))

    # Calculate confusion matrix
    cm = confusion_matrix(all_truths, all_predictions, labels=labels)

    return cm, labels


def calculate_metrics(truths: List[int], predictions: List[int]) -> Dict[str, float]:
    """
    Calculate detection rate, Jaccard index, false-positive rate, and per-class precision.

    Args:
        truths: Ground truth labels
        predictions: Predicted labels

    Returns:
        Dictionary containing detection_rate, jaccard_index, false_positive_rate, and class precisions
    """
    truths_arr = np.array(truths)
    predictions_arr = np.array(predictions)

    # Detection Rate: Proportion of non-zero gestures correctly detected
    # True Positives: non-zero truth correctly predicted as non-zero
    non_zero_truth_mask = truths_arr != 0
    if non_zero_truth_mask.sum() == 0:
        detection_rate = 1.0  # No gestures to detect
    else:
        correct_detections = (
            non_zero_truth_mask & (predictions_arr == truths_arr)
        ).sum()
        detection_rate = correct_detections / non_zero_truth_mask.sum()

    # Jaccard Index (IoU): Intersection over Union for each gesture class
    # Calculate per-class then average
    unique_labels = set(truths_arr) | set(predictions_arr)
    unique_labels.discard(0)  # Exclude background class

    if len(unique_labels) == 0:
        jaccard_index = 1.0  # Only background, perfect match
    else:
        jaccard_scores = []
        for label in unique_labels:
            truth_mask = truths_arr == label
            pred_mask = predictions_arr == label
            intersection = (truth_mask & pred_mask).sum()
            union = (truth_mask | pred_mask).sum()
            if union > 0:
                jaccard_scores.append(intersection / union)
        jaccard_index = np.mean(jaccard_scores) if jaccard_scores else 0.0

    # False Positive Rate: Proportion of zero truths incorrectly predicted as non-zero
    zero_truth_mask = truths_arr == 0
    if zero_truth_mask.sum() == 0:
        false_positive_rate = 0.0  # No background frames
    else:
        false_positives = ((predictions_arr != 0) & zero_truth_mask).sum()
        false_positive_rate = false_positives / zero_truth_mask.sum()

    # Per-class precision: For each class, calculate precision (TP / (TP + FP))
    # Precision = correct predictions for that class / total predictions for that class
    all_labels = set(truths_arr) | set(predictions_arr)
    class_precisions = {}

    for label in all_labels:
        pred_mask = predictions_arr == label
        if pred_mask.sum() == 0:
            # Class not predicted at all
            class_precisions[f"class_{label}_precision"] = None
        else:
            correct = (predictions_arr == label) & (truths_arr == label)
            class_precisions[f"class_{label}_precision"] = (
                correct.sum() / pred_mask.sum()
            )

    result = {
        "detection_rate": detection_rate,
        "jaccard_index": jaccard_index,
        "false_positive_rate": false_positive_rate,
    }
    result.update(class_precisions)

    return result


def evaluate_model(df: pd.DataFrame, model_name: str) -> Dict[str, float]:
    """
    Evaluate a single model across all sequences.

    Args:
        df: DataFrame filtered for a specific model
        model_name: Name of the model

    Returns:
        Dictionary containing averaged metrics
    """
    all_metrics = []

    for _, row in df.iterrows():
        truths = ast.literal_eval(row["truths"])
        predictions = ast.literal_eval(row["predictions"])
        metrics = calculate_metrics(truths, predictions)
        all_metrics.append(metrics)

    # Collect all class accuracy keys
    all_class_keys = set()
    for m in all_metrics:
        all_class_keys.update([k for k in m.keys() if k.startswith("class_")])

    # Average across all sequences
    avg_metrics = {
        "model": model_name,
        "detection_rate": np.mean([m["detection_rate"] for m in all_metrics]),
        "jaccard_index": np.mean([m["jaccard_index"] for m in all_metrics]),
        "false_positive_rate": np.mean([m["false_positive_rate"] for m in all_metrics]),
        "num_sequences": len(all_metrics),
    }

    # Average per-class accuracies (excluding None values)
    for class_key in sorted(all_class_keys):
        values = [m.get(class_key) for m in all_metrics if m.get(class_key) is not None]
        if values:
            avg_metrics[class_key] = np.mean(values)
        else:
            avg_metrics[class_key] = None

    return avg_metrics


def main(csv_path: str):
    """
    Main evaluation function.

    Args:
        csv_path: Path to the CSV file containing results
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Get unique models
    models = df["model"].unique()

    # Evaluate each model
    results = []
    for model in models:
        model_df = df[df["model"] == model]
        metrics = evaluate_model(model_df, model)
        results.append(metrics)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by detection rate (descending)
    results_df = results_df.sort_values("detection_rate", ascending=False)

    # Print formatted table - Overall metrics
    print("\n" + "=" * 120)
    print("GESTURE RECOGNITION EVALUATION RESULTS - OVERALL METRICS")
    print("=" * 120)
    print(
        f"\n{'Model':<30} {'Detection Rate':<20} {'Jaccard Index':<20} {'False Positive':<20} {'Sequences':<10}"
    )
    print("-" * 120)

    for _, row in results_df.iterrows():
        print(
            f"{row['model']:<30} {row['detection_rate']:<20.4f} {row['jaccard_index']:<20.4f} "
            f"{row['false_positive_rate']:<20.4f} {int(row['num_sequences']):<10}"
        )

    print("=" * 120 + "\n")

    # Print per-class precision table
    class_columns = [col for col in results_df.columns if col.startswith("class_")]
    if class_columns:
        # Extract class numbers and create gesture names mapping
        gesture_names = {
            0: "NONE",
            1: "GRAB",
            2: "PINCH",
            3: "TAP",
            4: "DENY",
            5: "KNOB",
            6: "EXPAND",
        }

        print("\n" + "=" * 120)
        print("PER-CLASS PRECISION")
        print("=" * 120)

        # Header
        header = f"\n{'Model':<30}"
        for col in sorted(class_columns):
            class_num = int(col.split("_")[1])
            gesture_name = gesture_names.get(class_num, f"Class{class_num}")
            header += f" {gesture_name:<15}"
        print(header)
        print("-" * 120)

        # Data rows
        for _, row in results_df.iterrows():
            line = f"{row['model']:<30}"
            for col in sorted(class_columns):
                val = row[col]
                if val is None or pd.isna(val):
                    line += f" {'N/A':<15}"
                else:
                    line += f" {val:<15.4f}"
            print(line)

        print("=" * 120 + "\n")

    # Generate confusion matrix visualizations
    gesture_names = {
        0: "NONE",
        1: "GRAB",
        2: "PINCH",
        3: "TAP",
        4: "DENY",
        5: "KNOB",
        6: "EXPAND",
    }

    num_models = len(models)
    # Calculate grid dimensions (prefer wider layout)
    cols = min(3, num_models)
    rows = (num_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if num_models == 1:
        axes = np.array([axes])
    axes = axes.flatten() if num_models > 1 else axes

    for idx, model in enumerate(models):
        model_df = df[df["model"] == model]
        cm, labels = calculate_confusion_matrix(model_df)

        # Create label names for the confusion matrix
        label_names = [gesture_names.get(label, f"Class{label}") for label in labels]

        # Plot confusion matrix
        ax = axes[idx] if num_models > 1 else axes[0]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
            ax=ax,
            cbar=True,
        )
        ax.set_title(f"Model: {model}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)

    # Hide extra subplots if any
    for idx in range(num_models, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
    print(f"Confusion matrices saved to confusion_matrices.png\n")

    return results_df


if __name__ == "__main__":
    csv_path = "./out/testing_results.csv"
    results_df = main(csv_path)

    # Optionally save results
    results_df.to_csv("evaluation_results.csv", index=False)
    print(f"Results saved to evaluation_results.csv")
