import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def parse_result_file(filepath):
    """Parse the result.md file to extract model results."""
    with open(filepath, "r") as f:
        content = f.read()

    # Split content by model sections
    models = {}

    # Define model sections
    sections = [
        ("GTCN Model", "GTCN"),
        ("Double-Head GTCN Model", "Double-Head"),
        ("Probability Threshold GTCN Model", "P-Threshold"),
    ]

    for section_name, model_key in sections:
        # Find the section - match until next single # heading or end of file
        pattern = f"# {section_name}(.*?)(?=\n# [^#]|$)"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            section_content = match.group(1)
            models[model_key] = parse_model_section(section_content)

    return models


def parse_model_section(content):
    """Parse a single model section."""
    model_data = {}

    # Extract training losses
    loss_pattern = r"Epoch \[(\d+)/\d+\], Loss: ([\d.]+)"
    losses = re.findall(loss_pattern, content)
    model_data["epochs"] = [int(e) for e, _ in losses]
    model_data["losses"] = [float(l) for _, l in losses]

    # Extract confusion matrix
    cm_pattern = r"Confusion Matrix:\n(\[\[.*?\]\])"
    cm_match = re.search(cm_pattern, content, re.DOTALL)
    if cm_match:
        cm_text = cm_match.group(1)
        # Parse matrix rows - find all rows with brackets
        rows = re.findall(r"\[([^\[\]]+)\]", cm_text)
        confusion_matrix = []
        for row in rows:
            # Split by whitespace and convert to integers
            values = [int(x.strip()) for x in row.split() if x.strip()]
            if values:  # Only add non-empty rows
                confusion_matrix.append(values)
        if confusion_matrix:
            model_data["confusion_matrix"] = np.array(confusion_matrix)

    # Extract classification metrics
    metrics_pattern = r"(NONE|GRAB|PINCH|TAP|DENY|KNOB|EXPAND)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)"
    metrics = re.findall(metrics_pattern, content)

    if metrics:
        model_data["classes"] = [m[0] for m in metrics]
        model_data["precision"] = [float(m[1]) for m in metrics]
        model_data["recall"] = [float(m[2]) for m in metrics]
        model_data["f1_score"] = [float(m[3]) for m in metrics]
        model_data["support"] = [int(m[4]) for m in metrics]

    # Extract overall accuracy
    accuracy_pattern = r"accuracy\s+([\d.]+)"
    accuracy_match = re.search(accuracy_pattern, content)
    if accuracy_match:
        model_data["accuracy"] = float(accuracy_match.group(1))

    # Extract training time
    time_pattern = r"Training completed in ([\d.]+) seconds"
    time_match = re.search(time_pattern, content)
    if time_match:
        model_data["training_time"] = float(time_match.group(1))

    return model_data


def plot_training_losses(models, save_path="training_losses.png"):
    """Plot training loss comparison between models."""
    plt.figure(figsize=(12, 6))

    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    for (model_name, model_data), color in zip(models.items(), colors):
        if "epochs" in model_data and "losses" in model_data:
            plt.plot(
                model_data["epochs"],
                model_data["losses"],
                label=model_name,
                linewidth=2,
                color=color,
                alpha=0.8,
            )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss Comparison", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_confusion_matrices(models, save_path="confusion_matrices.png"):
    """Plot confusion matrices for all models."""
    n_models = len([m for m in models.values() if "confusion_matrix" in m])

    fig, axes = plt.subplots(1, n_models, figsize=(18, 5))
    if n_models == 1:
        axes = [axes]

    class_names = ["NONE", "GRAB", "PINCH", "TAP", "DENY", "KNOB", "EXPAND"]

    for idx, (model_name, model_data) in enumerate(models.items()):
        if "confusion_matrix" in model_data:
            cm = model_data["confusion_matrix"]

            # Normalize confusion matrix
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[idx],
                cbar_kws={"label": "Normalized Value"},
            )

            axes[idx].set_title(
                f'{model_name}\nAccuracy: {model_data.get("accuracy", 0):.4f}',
                fontsize=12,
                fontweight="bold",
            )
            axes[idx].set_xlabel("Predicted", fontsize=10)
            axes[idx].set_ylabel("Actual", fontsize=10)
            axes[idx].tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_metrics_comparison(models, save_path="metrics_comparison.png"):
    """Plot comparison of precision, recall, and F1-score across models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    class_names = ["NONE", "GRAB", "PINCH", "TAP", "DENY", "KNOB", "EXPAND"]
    x = np.arange(len(class_names))
    width = 0.25

    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    # Plot Precision
    ax = axes[0, 0]
    for idx, (model_name, model_data) in enumerate(models.items()):
        if "precision" in model_data:
            offset = width * (idx - 1)
            ax.bar(
                x + offset,
                model_data["precision"],
                width,
                label=model_name,
                color=colors[idx],
                alpha=0.8,
            )
    ax.set_xlabel("Gesture Class", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Plot Recall
    ax = axes[0, 1]
    for idx, (model_name, model_data) in enumerate(models.items()):
        if "recall" in model_data:
            offset = width * (idx - 1)
            ax.bar(
                x + offset,
                model_data["recall"],
                width,
                label=model_name,
                color=colors[idx],
                alpha=0.8,
            )
    ax.set_xlabel("Gesture Class", fontsize=11)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title("Recall Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Plot F1-Score
    ax = axes[1, 0]
    for idx, (model_name, model_data) in enumerate(models.items()):
        if "f1_score" in model_data:
            offset = width * (idx - 1)
            ax.bar(
                x + offset,
                model_data["f1_score"],
                width,
                label=model_name,
                color=colors[idx],
                alpha=0.8,
            )
    ax.set_xlabel("Gesture Class", fontsize=11)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title("F1-Score Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Plot Overall Metrics
    ax = axes[1, 1]
    model_names = list(models.keys())
    accuracies = [models[m].get("accuracy", 0) for m in model_names]

    # Calculate macro F1
    macro_f1s = [np.mean(models[m].get("f1_score", [])) for m in model_names]

    x_pos = np.arange(len(model_names))
    width = 0.35

    ax.bar(
        x_pos - width / 2,
        accuracies,
        width,
        label="Accuracy",
        color="#2E86AB",
        alpha=0.8,
    )
    ax.bar(
        x_pos + width / 2,
        macro_f1s,
        width,
        label="Macro F1-Score",
        color="#A23B72",
        alpha=0.8,
    )

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Overall Performance Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1])

    # Add value labels on bars
    for i, (acc, f1) in enumerate(zip(accuracies, macro_f1s)):
        ax.text(
            i - width / 2,
            acc + 0.02,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            i + width / 2, f1 + 0.02, f"{f1:.3f}", ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_training_efficiency(models, save_path="training_efficiency.png"):
    """Plot training time vs accuracy."""
    plt.figure(figsize=(10, 6))

    model_names = list(models.keys())
    training_times = [models[m].get("training_time", 0) for m in model_names]
    accuracies = [models[m].get("accuracy", 0) for m in model_names]

    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    for i, (name, time, acc) in enumerate(zip(model_names, training_times, accuracies)):
        plt.scatter(
            time,
            acc,
            s=300,
            color=colors[i],
            alpha=0.7,
            edgecolors="black",
            linewidth=2,
            label=name,
        )
        plt.annotate(
            name,
            (time, acc),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc=colors[i], alpha=0.3),
        )

    plt.xlabel("Training Time (seconds)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Training Efficiency: Time vs Accuracy", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def print_summary_table(models):
    """Print a summary table of model performance."""
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 80)
    print(
        f"{'Model':<20} {'Accuracy':<12} {'Macro F1':<12} {'Training Time (s)':<18} {'Final Loss':<12}"
    )
    print("-" * 80)

    for model_name, model_data in models.items():
        accuracy = model_data.get("accuracy", 0)
        macro_f1 = np.mean(model_data.get("f1_score", [0]))
        training_time = model_data.get("training_time", 0)
        final_loss = (
            model_data.get("losses", [0])[-1] if model_data.get("losses") else 0
        )

        print(
            f"{model_name:<20} {accuracy:<12.4f} {macro_f1:<12.4f} {training_time:<18.2f} {final_loss:<12.4f}"
        )

    print("=" * 80)

    # Per-class performance
    print("\nPER-CLASS F1-SCORE COMPARISON")
    print("=" * 80)
    class_names = ["NONE", "GRAB", "PINCH", "TAP", "DENY", "KNOB", "EXPAND"]

    print(f"{'Class':<12}", end="")
    for model_name in models.keys():
        print(f"{model_name:<20}", end="")
    print()
    print("-" * 80)

    for i, class_name in enumerate(class_names):
        print(f"{class_name:<12}", end="")
        for model_name, model_data in models.items():
            f1_scores = model_data.get("f1_score", [])
            if i < len(f1_scores):
                print(f"{f1_scores[i]:<20.4f}", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()

    print("=" * 80 + "\n")


def main():
    """Main function to run all visualizations."""
    result_file = Path("result.md")

    if not result_file.exists():
        print(f"Error: {result_file} not found!")
        return

    print("Parsing result.md file...")
    models = parse_result_file(result_file)

    if not models:
        print("Error: No model data found in result.md")
        return

    print(f"Found {len(models)} models: {', '.join(models.keys())}\n")

    # Create visualizations
    print("Generating visualizations...")
    plot_training_losses(models, "training_losses.png")
    plot_confusion_matrices(models, "confusion_matrices.png")
    plot_metrics_comparison(models, "metrics_comparison.png")
    plot_training_efficiency(models, "training_efficiency.png")

    # Print summary
    print_summary_table(models)

    print("\n✓ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - training_losses.png")
    print("  - confusion_matrices.png")
    print("  - metrics_comparison.png")
    print("  - training_efficiency.png")


if __name__ == "__main__":
    main()
