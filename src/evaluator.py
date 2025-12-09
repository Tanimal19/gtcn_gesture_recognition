import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# Config
# -----------------------------
sequences = 72
min_overlap_ratio = 0.5
num_gest = 17

gestures = [
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "OK",
    "MENU",
    "LEFT",
    "RIGHT",
    "CIRCLE",
    "V",
    "CROSS",
    "GRAB",
    "PINCH",
    "TAP",
    "DENY",
    "KNOB",
    "EXPAND",
]


# -----------------------------
# Helper function to read annotations/results
# -----------------------------
def read_annotations(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(";")
            seq_id = int(parts[0])
            gestures_list = parts[1:-1]  # exclude last empty
            data.append([seq_id, gestures_list])
    return data


# -----------------------------
# Load results and annotations
# -----------------------------
results = read_annotations("annotations2.txt")
annotations = read_annotations("annotations.txt")
annotations.sort(key=lambda x: x[0])  # sort by sequence id

# -----------------------------
# Initialize metrics
# -----------------------------
confmat = np.zeros((18, 18), dtype=int)  # 17 gestures + 1 non-gesture class
class_results = np.zeros(
    (num_gest, 6), dtype=float
)  # Total, Correct, Missed, Misclassified, FalsePositive, Jaccard
jaccard_counts = np.zeros(num_gest)

# -----------------------------
# Main evaluation loop
# -----------------------------
for s in range(sequences):
    A = annotations[s][1]
    R = results[s][1]

    if len(R) == 0:
        for i in range(0, len(A), 3):
            label = A[i]
            idx = gestures.index(label)
            class_results[idx, 0] += 1

    found = np.zeros(len(A) // 3)

    for r in range(0, len(R), 3):
        RR = R[r : r + 3]
        detected = False
        countA = 0

        for a in range(0, len(A), 3):
            AA = A[a : a + 3]
            if r == 0:
                idx_A = gestures.index(AA[0])
                class_results[idx_A, 0] += 1  # Total

            countA += 1

            AA_len = int(AA[2]) - int(AA[1])
            overlap = min(int(AA[2]), int(RR[2])) - max(int(AA[1]), int(RR[1]))
            overlap_ratio = overlap / AA_len if AA_len > 0 else 0

            # Jaccard index
            if overlap > 0 and RR[0] == AA[0]:
                U = max(int(AA[2]), int(RR[2])) - min(int(AA[1]), int(RR[1]))
                idx_g = gestures.index(AA[0])
                class_results[idx_g, 5] += overlap / U
                jaccard_counts[idx_g] += 1

            if overlap_ratio > min_overlap_ratio:
                detected = True
                idx_AA = gestures.index(AA[0])

                if RR[0] == AA[0]:
                    if found[countA - 1] != 1:
                        idx_RR = gestures.index(RR[0])
                        confmat[idx_RR, idx_AA] += 1
                        found[countA - 1] = 1
                        class_results[idx_AA, 1] += 1  # Correct
                    else:
                        idx_RR = 17  # Non-gesture row
                        confmat[idx_RR, idx_AA] += 1
                else:
                    class_results[idx_AA, 3] += 1  # Misclassified
                    if found[countA - 1] != 1:
                        idx_RR = gestures.index(RR[0])
                        confmat[idx_RR, idx_AA] += 1
                    else:
                        idx_RR = 17
                        confmat[idx_RR, idx_AA] += 1

        if not detected:
            idx_AA = gestures.index(AA[0])
            class_results[idx_AA, 4] += 1  # False positive
            confmat[17, idx_AA] += 1  # Non-gesture row

    for f, val in enumerate(found):
        if val == 0:
            idx_AA = gestures.index(A[f * 3])
            class_results[idx_AA, 2] += 1  # Missed
            confmat[idx_AA, 17] += 1  # Non-gesture col

# -----------------------------
# Final metrics
# -----------------------------
for i in range(num_gest):
    class_results[i, 5] = class_results[i, 5] / (
        jaccard_counts[i]
        + class_results[i, 2]
        + class_results[i, 3]
        + class_results[i, 4]
    )

class_precision = class_results[:, 1] / (
    class_results[:, 1] + class_results[:, 2] + class_results[:, 4]
)
class_recall = class_results[:, 1] / (class_results[:, 1] + class_results[:, 3])

correct_score = np.sum(class_results[:, 1]) / np.sum(class_results[:, 0])
misclassified_rate = np.sum(class_results[:, 3]) / np.sum(class_results[:, 0])
false_positive_rate = np.sum(class_results[:, 4]) / np.sum(class_results[:, 0])

results_compact = np.column_stack(
    [
        class_results[:, 1] / 16,
        (class_results[:, 3] + class_results[:, 4]) / 16,
        class_results[:, 1]
        / (
            class_results[:, 1]
            + class_results[:, 2]
            + class_results[:, 3]
            + class_results[:, 4]
        ),
    ]
)
results_compact = np.vstack(
    [
        results_compact,
        [
            np.sum(results_compact[:, 0]) / 17,
            np.sum(results_compact[:, 1]) / 17,
            np.nansum(results_compact[:, 2]) / 17,
        ],
    ]
)

# -----------------------------
# Confusion matrix display
# -----------------------------
ge = gestures + ["NONGESTURES"]
disp = ConfusionMatrixDisplay(confmat, display_labels=ge)
disp.plot(xticks_rotation=90)
