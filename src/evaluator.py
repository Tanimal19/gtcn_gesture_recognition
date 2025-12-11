import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from src.dataset_utils import (
    SHREC_TEST_DATASET_FOLDER,
    GestureLabel,
    parse_shrec_annotations_file,
)
import argparse

num_sequence = 72
min_overlap_ratio = 0.5
gestures = [  # only dynamic gestures
    GestureLabel.GRAB,
    GestureLabel.PINCH,
    GestureLabel.TAP,
    GestureLabel.DENY,
    GestureLabel.KNOB,
    GestureLabel.EXPAND,
]
num_gest = len(gestures)
none_gest_idx = num_gest


def read_annotations(filepath):
    anns = parse_shrec_annotations_file(filepath, gestures)

    data = []
    for ann in anns:
        gestures_list = []
        for gesture in ann.gestures:
            gestures_list.extend([gesture[0], gesture[1], gesture[2]])
        data.append([ann.sequence_id, gestures_list])

    data.sort(key=lambda x: x[0])  # sort by sequence ID
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate generated annotation with ground truth."
    )
    parser.add_argument(
        "--ann",
        type=str,
        help="path of generated annotation file.",
    )
    args = parser.parse_args()

    # load annotations
    prediction = read_annotations(args.ann)
    truth = read_annotations(SHREC_TEST_DATASET_FOLDER + "annotations_revised.txt")

    # init metrics
    confmat = np.zeros((num_gest + 1, num_gest + 1), dtype=int)
    class_results = np.zeros(
        (num_gest, 6), dtype=float
    )  # Total, Correct, Missed, Misclassified, FalsePositive, Jaccard
    jaccard_counts = np.zeros(num_gest)

    # evaluate each sequence
    for s in range(num_sequence):
        A = truth[s][1]
        R = prediction[s][1]

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
                            # Non-gesture row
                            confmat[none_gest_idx, idx_AA] += 1
                    else:
                        class_results[idx_AA, 3] += 1  # Misclassified
                        if found[countA - 1] != 1:
                            idx_RR = gestures.index(RR[0])
                            confmat[idx_RR, idx_AA] += 1
                        else:
                            confmat[none_gest_idx, idx_AA] += 1

            if not detected:
                idx_AA = gestures.index(AA[0])
                class_results[idx_AA, 4] += 1  # False positive
                confmat[none_gest_idx, idx_AA] += 1  # Non-gesture row

        for f, val in enumerate(found):
            if val == 0:
                idx_AA = gestures.index(A[f * 3])
                class_results[idx_AA, 2] += 1  # Missed
                confmat[idx_AA, none_gest_idx] += 1  # Non-gesture col

    # finalize metrics
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

    print("\n=== Per-class results ===")
    print(
        f"{'Gesture':<15} {'Total':<7} {'Correct':<9} {'Missed':<7} {'Misclassified':<15} {'FalsePositive':<15} {'Jaccard':<10} {'Precision':<10} {'Recall':<10}"
    )
    print(
        f"{'Overall':<15} {int(np.sum(class_results[:,0])):<7} {int(np.sum(class_results[:,1])):<9} {int(np.sum(class_results[:,2])):<7} {int(np.sum(class_results[:,3])):<15} {int(np.sum(class_results[:,4])):<15} {'-':<10} {'-':<10} {'-':<10}"
    )
    for i, g in enumerate(gestures):
        precision = class_precision[i] if not np.isnan(class_precision[i]) else 0.0
        recall = class_recall[i] if not np.isnan(class_recall[i]) else 0.0
        print(
            f"{g.name:<15} {int(class_results[i,0]):<7} {int(class_results[i,1]):<9} {int(class_results[i,2]):<7} {int(class_results[i,3]):<15} {int(class_results[i,4]):<15} {class_results[i,5]:<10.4f} {precision:<10.4f} {recall:<10.4f}"
        )

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
                np.sum(results_compact[:, 0]) / num_gest,
                np.sum(results_compact[:, 1]) / num_gest,
                np.nansum(results_compact[:, 2]) / num_gest,
            ],
        ]
    )

    # -----------------------------
    # Confusion matrix display
    # -----------------------------
    ge = [g.name for g in gestures] + ["NONGESTURES"]
    disp = ConfusionMatrixDisplay(confmat, display_labels=ge)
    disp.plot(xticks_rotation=90)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
