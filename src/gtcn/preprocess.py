from src.dataset_utils import (
    parse_shrec_sequence_file,
    parse_shrec_annotations_file,
    DAnnotation,
    DSequenceFrame,
    SHREC_TRAINING_DATASET_FOLDER,
    SHREC_TEST_DATASET_FOLDER,
)
from src.gtcn.model import GTCNModel
from sklearn.model_selection import train_test_split
from typing import List
from collections import Counter
import torch
import numpy as np
import pickle
import os


DATASET_FOLDER = "./src/gtcn/datasets/"


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
    y = np.full(total_frames, -1, dtype=np.int32)
    for label, start_frame, end_frame in annotation.gestures:
        y[start_frame:end_frame] = label.value

    # output shape: (total_frames,)
    return y


def stratified_split(
    X: np.ndarray, y: np.ndarray, val_size=0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N = len(y)
    indices = list(range(N))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        stratify=y,
        random_state=42,
    )

    print("y distribution:")
    print(f"train: {Counter(y[train_idx])}")
    print(f"val: {Counter(y[val_idx])}")

    return (X[train_idx], y[train_idx], X[val_idx], y[val_idx])


def create_dataset(sequences_folder, ann_file, output_folder, max_sequence_id=None):
    os.makedirs(output_folder, exist_ok=True)

    annotations = parse_shrec_annotations_file(ann_file)
    for ann in annotations:
        if max_sequence_id is not None and ann.sequence_id > max_sequence_id:
            continue

        print(f"+ Processing sequence: {ann.sequence_id}")
        sequence_file = sequences_folder + str(ann.sequence_id) + ".txt"
        sequence = parse_shrec_sequence_file(sequence_file)

        # Convert each window in the sequence to X
        X: List[np.ndarray] = []
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

        print(f"X.shape: {np.array(X).shape}, y.shape: {y.shape}")

        X_train, y_train, X_val, y_val = stratified_split(
            X=np.array(X), y=y, val_size=0.2
        )

        output_file = output_folder + f"s{ann.sequence_id}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(
                {
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_val": X_val,
                    "y_val": y_val,
                },
                f,
            )
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"-> Saved to {output_file} ({file_size:.2f} MB)\n")


if __name__ == "__main__":
    create_dataset(
        sequences_folder=SHREC_TRAINING_DATASET_FOLDER + "sequences/",
        ann_file=SHREC_TRAINING_DATASET_FOLDER + "annotations_revised_training.txt",
        output_folder=DATASET_FOLDER + "training_set/",
        max_sequence_id=5,
    )

    # create_dataset(
    #     sequences_folder=SHREC_TEST_DATASET_FOLDER + "sequences/",
    #     ann_file=SHREC_TEST_DATASET_FOLDER + "annotations_revised.txt",
    #     output_folder=DATASET_FOLDER + "test/",
    # )
