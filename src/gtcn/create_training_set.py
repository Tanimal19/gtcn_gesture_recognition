from src.dataset_utils import (
    parse_shrec_sequence_file,
    parse_shrec_annotations_file,
    DAnnotation,
    DSequenceFrame,
    SHREC_TRAINING_DATASET_FOLDER,
)
from src.gtcn.model import GTCNModel
from typing import List
from collections import Counter
import numpy as np
import pickle
import os


def convert_window_to_X(window: List[DSequenceFrame]) -> np.ndarray:
    num_landmarks = len(GTCNModel.LANDMARKS)
    X = np.zeros((GTCNModel.WINDOW_LENGTH, num_landmarks, 3), dtype=np.float32)

    for t, frame in enumerate(window):
        for i, lm in enumerate(GTCNModel.LANDMARKS):
            coord = frame.landmarks[lm]
            X[t, i, :] = np.array(coord, dtype=np.float32)

    # output shape: (WINDOW_LENGTH, num_landmarks, 3)
    return X


def convert_annotation_to_y(annotation: DAnnotation, total_frames: int) -> np.ndarray:
    y = np.full(total_frames, -1, dtype=np.long)

    for label, start_frame, end_frame in annotation.gestures:
        pre_start = max(0, start_frame - GTCNModel.WINDOW_LENGTH)
        post_end = min(total_frames, end_frame + GTCNModel.WINDOW_LENGTH)
        y[pre_start:post_end] = 0  # non gesture
        y[start_frame:end_frame] = GTCNModel.GESTURES.index(label)

    # output shape: (total_frames,)
    return y


def create_training_set(
    sequences_folder, ann_file, output_folder, max_sequence_id=None
):
    os.makedirs(output_folder, exist_ok=True)

    total_X = np.array([], dtype=np.float32).reshape(
        0, GTCNModel.WINDOW_LENGTH, len(GTCNModel.LANDMARKS), 3
    )
    total_y = np.array([], dtype=np.long)
    total_seq_ids = np.array([], dtype=np.int32)  # Track sequence IDs

    annotations = parse_shrec_annotations_file(ann_file, GTCNModel.GESTURES)
    for ann in annotations:
        if max_sequence_id is not None and ann.sequence_id > max_sequence_id:
            continue

        print(f"+ Processing sequence: {ann.sequence_id}")
        sequence_file = sequences_folder + str(ann.sequence_id) + ".txt"
        sequence = parse_shrec_sequence_file(sequence_file)

        # Convert each window in the sequence to X
        X = []
        for end_frame in range(len(sequence.frames)):
            start_frame = end_frame - GTCNModel.WINDOW_LENGTH + 1
            if start_frame >= 0:
                window = sequence.frames[start_frame : end_frame + 1]
            else:
                window = [sequence.frames[0]] * (-start_frame) + sequence.frames[
                    : end_frame + 1
                ]

            assert len(window) == GTCNModel.WINDOW_LENGTH
            X.append(convert_window_to_X(window))

        y = convert_annotation_to_y(ann, len(sequence.frames))

        mask = y != -1
        X = np.array(X)[mask]
        y = y[mask]
        seq_ids = np.full(
            len(y), ann.sequence_id, dtype=np.int32
        )  # Create sequence ID array

        total_X = np.concatenate((total_X, X), axis=0)
        total_y = np.concatenate((total_y, y), axis=0)
        total_seq_ids = np.concatenate((total_seq_ids, seq_ids), axis=0)

    print(f"X.shape: {total_X.shape}, y.shape: {total_y.shape}")
    c = Counter(total_y)
    distribution_str = "y label distribution:"
    for label_idx in range(len(GTCNModel.GESTURES)):
        distribution_str += f" {GTCNModel.GESTURES[label_idx].name}:{c[label_idx]}"
    print(distribution_str)

    output_file = output_folder + "train.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(
            {"X": total_X, "y": total_y, "seq_ids": total_seq_ids},
            f,
        )
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Saved to {output_file} ({file_size:.2f} MB)")


if __name__ == "__main__":
    create_training_set(
        sequences_folder=SHREC_TRAINING_DATASET_FOLDER + "sequences/",
        ann_file=SHREC_TRAINING_DATASET_FOLDER + "annotations_revised_training.txt",
        output_folder="./src/gtcn/datasets/",
        max_sequence_id=None,
    )
