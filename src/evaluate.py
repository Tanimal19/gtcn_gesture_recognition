import pandas as pd
import numpy as np
from typing import Any
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import argparse
from src.gtcn import OUTPUT_GESTURES


def calculate_sequence_metrics(
    truths: list[int], predictions: list[int]
) -> dict[str, float]:
    truths_arr = np.array(truths)
    predictions_arr = np.array(predictions)

    # True Positives: non-zero truth correctly predicted as non-zero
    non_zero_truth_mask = truths_arr != 0
    correct_detections = (non_zero_truth_mask & (predictions_arr == truths_arr)).sum()
    true_positive_rate = correct_detections / non_zero_truth_mask.sum()

    # False Positive Rate: Proportion of zero truths incorrectly predicted as non-zero
    zero_truth_mask = truths_arr == 0
    false_positives = ((predictions_arr != 0) & zero_truth_mask).sum()
    false_positive_rate = false_positives / zero_truth_mask.sum()

    # Jaccard Index (IoU): Intersection over Union for each gesture class
    unique_labels = set(truths_arr) | set(predictions_arr)
    unique_labels.discard(0)  # exclude None class
    jaccard_scores = []
    for label in unique_labels:
        truth_mask = truths_arr == label
        pred_mask = predictions_arr == label
        intersection = (truth_mask & pred_mask).sum()
        union = (truth_mask | pred_mask).sum()
        if union > 0:
            jaccard_scores.append(intersection / union)
    jaccard_index = np.mean(jaccard_scores) if jaccard_scores else 0.0

    result = {
        "tp": true_positive_rate,
        "fp": false_positive_rate,
        "ji": jaccard_index,
    }

    return result


def evaluate_model(
    model_name: str,
    sequences: pd.DataFrame,
    disable_metrics,
    disable_class_report,
    disable_confusion,
):
    print("\n" + "=" * 80)
    print(f"EVALUATION REPORT OF {model_name.upper()}")
    print("=" * 80)

    results: dict[str, Any] = {
        "model": model_name,
    }

    all_truths = []
    all_predictions = []
    for _, seq in sequences.iterrows():
        all_truths.append(ast.literal_eval(seq["truths"]))
        all_predictions.append(ast.literal_eval(seq["predictions"]))

    if not disable_metrics:
        all_metrics = []
        for truths, predictions in zip(all_truths, all_predictions):
            all_metrics.append(calculate_sequence_metrics(truths, predictions))

        results.update(
            {
                "true_positive": np.mean([m["tp"] for m in all_metrics]),
                "false_positive": np.mean([m["fp"] for m in all_metrics]),
                "jaccard_index": np.mean([m["ji"] for m in all_metrics]),
            }
        )
        print(
            f"True Positive Rate: {results['true_positive']:.4f}, "
            f"False Positive Rate: {results['false_positive']:.4f}, "
            f"Jaccard Index: {results['jaccard_index']:.4f}"
        )

    flat_truths = [item for sublist in all_truths for item in sublist]
    flat_predictions = [item for sublist in all_predictions for item in sublist]
    if not disable_class_report:
        class_report = classification_report(
            flat_truths,
            flat_predictions,
            zero_division=0,
            target_names=[g.name for g in OUTPUT_GESTURES],
        )
        print("Classification Report:")
        print(class_report)

    if not disable_confusion:
        cm = confusion_matrix(flat_truths, flat_predictions)
        print("Confusion Matrix:")
        print(cm)
        results["cm"] = cm.tolist()

    return results


def evaluate_csv(
    csv_path: str,
    disable_metrics=False,
    disable_class_report=False,
    disable_confusion=False,
):
    df = pd.read_csv(csv_path)
    models = df["model"].unique()

    results = []
    for model_name in models:
        model_df = df[df["model"] == model_name]
        results.append(
            evaluate_model(
                model_name,
                model_df,
                disable_metrics,
                disable_class_report,
                disable_confusion,
            )
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("jaccard_index", ascending=False)

    if not disable_metrics and len(results_df) > 1:
        print("\n" + "=" * 100)
        print("EVALUATION METRICS COMPARISON ACROSS MODELS")
        print("=" * 100)
        print(
            f"\n{'Model':<30} {'Jaccard Index':<20} {'True Positive': <20} {'False Positive':<20}"
        )
        print("-" * 100)
        for _, row in results_df.iterrows():
            print(
                f"{row['model']:<30} {row['jaccard_index']:<20.4f} {row['true_positive']:<20.4f} "
                f"{row['false_positive']:<20.4f}"
            )

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate gesture recognition model results from CSV file."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
    )
    parser.add_argument(
        "--no-class-report",
        action="store_true",
    )
    parser.add_argument(
        "--no-confusion-matrix",
        action="store_true",
    )

    args = parser.parse_args()

    results_df = evaluate_csv(
        csv_path=args.input,
        disable_metrics=args.no_metrics,
        disable_class_report=args.no_class_report,
        disable_confusion=args.no_confusion_matrix,
    )
