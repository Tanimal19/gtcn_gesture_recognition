import pickle
import os
import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from src.dataset_utils import (
    parse_shrec_sequence_file,
    parse_shrec_annotations_file,
    DAnnotation,
    DSequence,
    SHREC_TRAINING_DATASET_FOLDER,
    SHREC_TEST_DATASET_FOLDER,
)
from src.gtcn import DEFAULT_TRAINSET_PATH, DEFAULT_TESTSET_PATH
from src.gtcn.model import GTCNModel


class GTCNDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.check_shape(X, y)
        print(f"> Creating GestureDataset with {y.shape[0]} samples.")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @staticmethod
    def check_shape(X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0], "X and y length mismatch!"
        assert X.shape[1] == GTCNModel.WINDOW_LENGTH, "X window length mismatch!"
        assert X.shape[2] == len(GTCNModel.LANDMARKS), "X landmark count mismatch!"
        assert X.shape[3] == 3, "X coordinate dimension mismatch!"

        y_unique = np.unique(y)
        for label in y_unique:
            assert 0 <= label < len(GTCNModel.GESTURES), "y label value out of range!"

    @staticmethod
    def print_label_distribution(y: np.ndarray):
        counts = Counter(y)
        print("> Label distribution:")
        for gesture in GTCNModel.GESTURES:
            idx = GTCNModel.GESTURES.index(gesture)
            print(f"  {gesture.name}:{counts[idx]};")


class GTCNDatasetBuilder:
    def __init__(self, shift: int = 0):
        self.shift = shift
        self.none_window_length = GTCNModel.WINDOW_LENGTH

    def convert_sequence_to_X(self, sequence: DSequence) -> np.ndarray:
        X = []
        for sample_frame in range(len(sequence.frames)):
            start_frame = sample_frame - GTCNModel.WINDOW_LENGTH + 1 + self.shift
            end_frame = start_frame + GTCNModel.WINDOW_LENGTH - 1

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

            assert len(window) == GTCNModel.WINDOW_LENGTH

            num_landmarks = len(GTCNModel.LANDMARKS)
            x = np.zeros((GTCNModel.WINDOW_LENGTH, num_landmarks, 3), dtype=np.float32)

            for t, frame in enumerate(window):
                for i, lm in enumerate(GTCNModel.LANDMARKS):
                    coord = frame.landmarks[lm]
                    x[t, i, :] = np.array(coord, dtype=np.float32)

            X.append(x)
        X = np.array(X)

        # output shape: (total_frames, WINDOW_LENGTH, num_landmarks, 3)
        return X

    def convert_annotation_to_y(
        self, annotation: DAnnotation, num_frames: int
    ) -> np.ndarray:
        y = np.full(num_frames, -1, dtype=np.long)

        for label, start_frame, end_frame in annotation.gestures:
            pre_start = max(0, start_frame - self.none_window_length)
            post_end = min(num_frames, end_frame + self.none_window_length)
            y[pre_start:post_end] = 0  # none gesture
            y[start_frame:end_frame] = GTCNModel.GESTURES.index(label)

        # output shape: (total_frames,)
        return y

    def create_set(
        self,
        sequences_folder,
        ann_file,
        out_file,
        max_sequence_id=None,
    ):
        total_X = np.array([], dtype=np.float32).reshape(
            0, GTCNModel.WINDOW_LENGTH, len(GTCNModel.LANDMARKS), 3
        )
        total_y = np.array([], dtype=np.long)
        total_seq_ids = np.array([], dtype=np.int32)  # track sequence IDs

        annotations = parse_shrec_annotations_file(ann_file, GTCNModel.GESTURES)
        for ann in annotations:
            if max_sequence_id is not None and ann.sequence_id > max_sequence_id:
                continue

            if len(ann.gestures) == 0:
                print(f"- Skipping sequence: {ann.sequence_id} (no available gestures)")
                continue

            print(f"+ Processing sequence: {ann.sequence_id}")
            sequence_file = sequences_folder + str(ann.sequence_id) + ".txt"
            sequence = parse_shrec_sequence_file(sequence_file)

            X = self.convert_sequence_to_X(sequence)
            y = self.convert_annotation_to_y(ann, len(sequence.frames))

            mask = y != -1
            X = np.array(X)[mask]
            y = y[mask]
            seq_ids = np.full(len(y), ann.sequence_id, dtype=np.int32)

            GTCNDataset.check_shape(X, y)

            total_X = np.concatenate((total_X, X), axis=0)
            total_y = np.concatenate((total_y, y), axis=0)
            total_seq_ids = np.concatenate((total_seq_ids, seq_ids), axis=0)

        GTCNDataset.check_shape(total_X, total_y)
        print(f"X.shape: {total_X.shape}, y.shape: {total_y.shape}")
        GTCNDataset.print_label_distribution(total_y)

        with open(out_file, "wb") as f:
            pickle.dump(
                {"X": total_X, "y": total_y, "seq_ids": total_seq_ids},
                f,
            )
        file_size = os.path.getsize(out_file) / (1024 * 1024)
        print(f"Saved to {out_file} ({file_size:.2f} MB)")


if __name__ == "__main__":
    builder = GTCNDatasetBuilder(shift=0)
    builder.create_set(
        sequences_folder=SHREC_TRAINING_DATASET_FOLDER + "sequences/",
        ann_file=SHREC_TRAINING_DATASET_FOLDER + "annotations_revised_training.txt",
        out_file=DEFAULT_TRAINSET_PATH,
    )
    builder.create_set(
        sequences_folder=SHREC_TEST_DATASET_FOLDER + "sequences/",
        ann_file=SHREC_TEST_DATASET_FOLDER + "annotations_revised.txt",
        out_file=DEFAULT_TESTSET_PATH,
    )
