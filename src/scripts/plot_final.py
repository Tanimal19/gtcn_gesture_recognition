import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the class labels
labels = ["NONE", "GRAB", "PINCH", "TAP", "DENY", "KNOB", "EXPAND"]

# Confusion matrices from final_result.txt
cm_final = np.array(
    [
        [1344, 399, 297, 395, 175, 91, 74],
        [104, 328, 10, 4, 0, 0, 7],
        [72, 14, 503, 5, 0, 10, 4],
        [89, 1, 45, 265, 21, 0, 0],
        [31, 0, 24, 147, 1045, 2, 0],
        [120, 139, 0, 11, 0, 934, 3],
        [114, 73, 16, 0, 0, 0, 155],
    ]
)

cm_final_peek1 = np.array(
    [
        [1482, 591, 78, 286, 118, 85, 135],
        [147, 298, 0, 0, 0, 0, 8],
        [126, 25, 448, 0, 0, 9, 0],
        [83, 6, 4, 319, 9, 0, 0],
        [48, 2, 8, 124, 1066, 0, 1],
        [292, 124, 0, 0, 0, 780, 11],
        [118, 67, 0, 0, 0, 0, 173],
    ]
)

cm_final_peek5 = np.array(
    [
        [2035, 172, 138, 206, 105, 50, 69],
        [195, 238, 0, 0, 0, 17, 3],
        [133, 0, 465, 9, 0, 1, 0],
        [159, 0, 6, 256, 0, 0, 0],
        [25, 0, 6, 45, 1173, 0, 0],
        [377, 82, 0, 3, 0, 732, 13],
        [160, 19, 0, 0, 0, 0, 179],
    ]
)

# Model names and metrics
models = [
    ("peek0", cm_final),
    ("peek1", cm_final_peek1),
    ("peek5", cm_final_peek5),
]

# Create figure with 3 subplots arranged horizontally
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot each confusion matrix
for idx, (model_name, cm) in enumerate(models):
    ax = axes[idx]

    # Normalize confusion matrix to percentages (by row)
    cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create heatmap
    sns.heatmap(
        cm_percentage,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        vmin=0,
        vmax=100,
    )

    # Set labels and title
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"{model_name}", fontsize=13)

    # Rotate x labels for better readability
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save figure
output_path = "data/final/confusion_matrices_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Confusion matrices plot saved to: {output_path}")

# Show plot
plt.show()
