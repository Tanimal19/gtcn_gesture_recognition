from src.dataset_utils import (
    parse_shrec_sequence_file,
    parse_shrec_annotations_file,
    DAnnotation,
    DSequenceFrame,
    SHREC_TRAINING_DATASET_FOLDER,
    SHREC_TEST_DATASET_FOLDER,
)
from src.gtcn.model import GTCNModel
from dataclasses import dataclass
from typing import List
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


def create_dataset(sequences_folder, ann_file, output_file, max_sequence_id=None):
    X: List[torch.Tensor] = []
    y = np.array([], dtype=np.int32)

    annotations = parse_shrec_annotations_file(ann_file)
    for ann in annotations:
        if max_sequence_id is not None and ann.sequence_id > max_sequence_id:
            continue

        print(f"+ Processing sequence: {ann.sequence_id}")
        sequence_file = sequences_folder + ann.sequence_id + ".txt"
        sequence = parse_shrec_sequence_file(sequence_file)

        # Convert each window in the sequence to X
        for end_frame in range(len(sequence.frames)):
            start_frame = end_frame - GTCNModel.WINDOW_LENGTH + 1
            if start_frame >= 0:
                window = sequence.frames[start_frame : end_frame + 1]
            else:
                window = [sequence.frames[0]] * (-start_frame) + sequence.frames[
                    : end_frame + 1
                ]

            assert len(window) == GTCNModel.WINDOW_LENGTH
            X_npy = convert_window_to_X(window)

            X.append(torch.tensor(X_npy, dtype=torch.float32))

        y_npy = convert_annotation_to_y(ann, len(sequence.frames))
        y = np.concatenate((y, y_npy), axis=0)

    print(f"X.shape: {(len(X),) + X[0].shape}, y.shape: {y.shape}")
    with open(output_file, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)

    file_size = os.path.getsize(output_file)
    print(f"Saved pickle file size: {file_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    # create_dataset(
    #     sequences_folder=SHREC_TRAINING_DATASET_FOLDER + "sequences/",
    #     ann_file=SHREC_TRAINING_DATASET_FOLDER + "annotations_revised_training.txt",
    #     output_file=DATASET_FOLDER + "training_set.pkl",
    # )

    create_dataset(
        sequences_folder=SHREC_TEST_DATASET_FOLDER + "sequences/",
        ann_file=SHREC_TEST_DATASET_FOLDER + "annotations_revised.txt",
        output_file=DATASET_FOLDER + "test_set.pkl",
    )
