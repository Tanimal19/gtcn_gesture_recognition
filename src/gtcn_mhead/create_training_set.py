from src.dataset_utils import (
    parse_shrec_sequence_file,
    parse_shrec_annotations_file,
    DAnnotation,
    DSequence,
    SHREC_TRAINING_DATASET_FOLDER,
)
from src.gtcn_mhead.model import GTCNMHead
from torch.utils.data import Dataset
from collections import Counter
import numpy as np
import pickle
import os
import torch

DEFAULT_TRAINSET_PATH = "./src/gtcn_mhead/datasets/train.pkl"


class TrainingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0], "X and y length mismatch!"
        assert X.shape[1] == GTCNMHead.WINDOW_LENGTH, "X window length mismatch!"
        assert X.shape[2] == len(GTCNMHead.LANDMARKS), "X landmark count mismatch!"
        assert X.shape[3] == 3, "X coordinate dimension mismatch!"

        print(f"> Creating TrainingDataset: X={X.shape}, y={y.shape}")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def convert_sequence_to_X(sequence: DSequence, shift: int = 0) -> np.ndarray:
    X = []
    for sample_frame in range(len(sequence.frames)):
        start_frame = sample_frame - GTCNMHead.WINDOW_LENGTH + 1 + shift
        end_frame = start_frame + GTCNMHead.WINDOW_LENGTH - 1

        if start_frame >= 0 and end_frame < len(sequence.frames):
            window = sequence.frames[start_frame : end_frame + 1]
        else:
            if end_frame >= len(sequence.frames):
                window = sequence.frames[start_frame:] + [sequence.frames[-1]] * (
                    end_frame - len(sequence.frames) + 1
                )
            elif start_frame < 0:
                window = [sequence.frames[0]] * (-start_frame) + sequence.frames[
                    : end_frame + 1
                ]
            else:
                raise ValueError("Unexpected frame indices!")

        assert len(window) == GTCNMHead.WINDOW_LENGTH

        num_landmarks = len(GTCNMHead.LANDMARKS)
        x = np.zeros((GTCNMHead.WINDOW_LENGTH, num_landmarks, 3), dtype=np.float32)

        for t, frame in enumerate(window):
            for i, lm in enumerate(GTCNMHead.LANDMARKS):
                coord = frame.landmarks[lm]
                x[t, i, :] = np.array(coord, dtype=np.float32)

        X.append(x)

    X = np.array(X)

    # output shape: (total_frames, WINDOW_LENGTH, num_landmarks, 3)
    assert X.shape[0] == len(sequence.frames), "X length mismatch!"
    assert X.shape[1] == GTCNMHead.WINDOW_LENGTH, "X window length mismatch!"
    assert X.shape[2] == len(GTCNMHead.LANDMARKS), "X landmark count mismatch!"
    assert X.shape[3] == 3, "X coordinate dimension mismatch!"

    return X


def convert_annotation_to_y(annotation: DAnnotation, num_frames: int) -> np.ndarray:
    y = np.full(num_frames, -1, dtype=np.long)

    for label, start_frame, end_frame in annotation.gestures:
        pre_start = max(0, start_frame - GTCNMHead.WINDOW_LENGTH)
        post_end = min(num_frames, end_frame + GTCNMHead.WINDOW_LENGTH)
        y[pre_start:post_end] = 0  # non gesture
        y[start_frame:end_frame] = GTCNMHead.GESTURES.index(label)

    # output shape: (total_frames,)
    assert y.shape[0] == num_frames, "y length mismatch!"

    return y


def create_training_set(
    sequences_folder,
    ann_file,
    out_file=DEFAULT_TRAINSET_PATH,
    shift=0,
    max_sequence_id=None,
):
    total_X = np.array([], dtype=np.float32).reshape(
        0, GTCNMHead.WINDOW_LENGTH, len(GTCNMHead.LANDMARKS), 3
    )
    total_y = np.array([], dtype=np.long)
    total_seq_ids = np.array([], dtype=np.int32)  # Track sequence IDs

    annotations = parse_shrec_annotations_file(ann_file, GTCNMHead.GESTURES)
    for ann in annotations:
        if max_sequence_id is not None and ann.sequence_id > max_sequence_id:
            continue

        print(f"+ Processing sequence: {ann.sequence_id}")
        sequence_file = sequences_folder + str(ann.sequence_id) + ".txt"
        sequence = parse_shrec_sequence_file(sequence_file)

        X = convert_sequence_to_X(sequence, shift)
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
        print(f"  -> samples added: {len(y)}, {Counter(y)}")

    print(f"X.shape: {total_X.shape}, y.shape: {total_y.shape}")
    c = Counter(total_y)
    distribution_str = "y label distribution:"
    for label_idx in range(len(GTCNMHead.GESTURES)):
        distribution_str += f" {GTCNMHead.GESTURES[label_idx].name}:{c[label_idx]}"
    print(distribution_str)

    with open(out_file, "wb") as f:
        pickle.dump(
            {"X": total_X, "y": total_y, "seq_ids": total_seq_ids},
            f,
        )
    file_size = os.path.getsize(out_file) / (1024 * 1024)
    print(f"Saved to {out_file} ({file_size:.2f} MB)")


if __name__ == "__main__":
    create_training_set(
        sequences_folder=SHREC_TRAINING_DATASET_FOLDER + "sequences/",
        ann_file=SHREC_TRAINING_DATASET_FOLDER + "annotations_revised_training.txt",
        max_sequence_id=None,
    )
