import numpy as np
from src.utils import (
    GestureLabel,
    SHREC_TRAINING_DATASET_FOLDER,
    parse_shrec_annotations_file,
    parse_shrec_sequence_file,
)


if __name__ == "__main__":
    sequences_folder = SHREC_TRAINING_DATASET_FOLDER + "sequences/"
    ann_file = SHREC_TRAINING_DATASET_FOLDER + "annotations_revised_training.txt"

    annotations = parse_shrec_annotations_file(ann_file)
    gesture_occurrences = {
        g: [] for g in GestureLabel
    }  # g: [consecutive frames 1, consecutive frames 2, ...]

    for ann in annotations:
        seq_file = sequences_folder + str(ann.sequence_id) + ".txt"
        sequence = parse_shrec_sequence_file(seq_file)

        gesture_frames = 0
        for label, start_frame, end_frame in ann.gestures:
            gesture_occurrences[label].append(end_frame - start_frame)
            gesture_frames += end_frame - start_frame

        gesture_occurrences[GestureLabel.NONE].append(
            len(sequence.frames) - gesture_frames
        )

    print(
        f"\n{'Gesture':<20} {'Total Occurrences':<20} {'Total Frames':<15} {'Avg Duration':<15} {'Std Duration':<15}"
    )
    for g in GestureLabel:
        occurrences = gesture_occurrences[g]
        total_occurrences = len(occurrences)
        total_frames = sum(occurrences)
        avg_duration = np.mean(occurrences)
        std_duration = np.std(occurrences)
        print(
            f"{g.name:<20} {total_occurrences:<20} {total_frames:<15} {avg_duration:<15.2f} {std_duration:<15.2f}"
        )
