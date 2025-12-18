import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Read the training history file
with open("./data/arch_comparison/training_history.txt", "r") as f:
    lines = f.readlines()

# Parse the data
models = {}
for line in lines:
    if ":" in line:
        model_name, loss_values = line.split(":", 1)
        model_name = model_name.strip()
        loss_values = ast.literal_eval(loss_values.strip())
        models[model_name] = loss_values

# Create the plot
plt.figure(figsize=(14, 8))

# Plot each model's training loss
for model_name, loss_values in models.items():
    epochs = range(1, len(loss_values) + 1)
    line = plt.plot(epochs, loss_values, label=model_name, linewidth=2, alpha=0.8)

    # Find best loss and its epoch
    best_loss = min(loss_values)
    best_epoch = loss_values.index(best_loss) + 1  # +1 because epochs start at 1

    # Add label at the position of best loss
    color = line[0].get_color()
    plt.text(
        best_epoch,
        best_loss if model_name != "win30" else best_loss - 0.03,
        f" {model_name}: {best_loss:.4f}",
        verticalalignment="center",
        fontsize=10,
        color=color,
        weight="bold",
        alpha=0.9,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.8,
            linewidth=1.5,
        ),
    )

# Customize the plot
plt.xlabel("Epoch", fontsize=14, fontweight="bold")
plt.ylabel("Training Loss", fontsize=14, fontweight="bold")
plt.legend(loc="upper right", fontsize=10, frameon=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig("training_loss_comparison.png", dpi=300, bbox_inches="tight")

# Display the plot
plt.show()

# Print summary statistics
print("\n=== Training Loss Summary ===")
print(
    f"{'Model':<30} {'Initial Loss':>12} {'Final Loss':>12} {'Best Loss':>12} {'Epochs':>8}"
)
print("=" * 80)
for model_name, loss_values in models.items():
    initial = loss_values[0]
    final = loss_values[-1]
    best = min(loss_values)
    epochs = len(loss_values)
    print(f"{model_name:<30} {initial:>12.4f} {final:>12.4f} {best:>12.4f} {epochs:>8}")
