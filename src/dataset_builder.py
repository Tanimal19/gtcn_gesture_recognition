import pickle
import os
import numpy as np
from src.utils import (
    parse_shrec_sequence_file,
    parse_shrec_annotations_file,
    DAnnotation,
    DSequence,
    SHREC_TRAINING_DATASET_FOLDER,
    SHREC_TEST_DATASET_FOLDER,
)
from src.gtcn import INPUT_LANDMARKS, OUTPUT_GESTURES, GTCNDataset


DEFAULT_DATASET_FOLDER = "./src/datasets/"


class GTCNDatasetBuilder:
    """
    :param window_length: number of frames in each input window
    :param peek: number of future frames to peek when constructing the window (default: 0)
    """

    def __init__(self, window_length, peek: int = 0):
        self.window_length = window_length
        self.none_window_length = window_length
        self.peek = peek

    def convert_sequence_to_X(self, sequence: DSequence) -> np.ndarray:
        X = []
        for sample_frame in range(len(sequence.frames)):
            start_frame = sample_frame - self.window_length + 1 + self.peek
            end_frame = start_frame + self.window_length - 1

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

            assert len(window) == self.window_length

            num_landmarks = len(INPUT_LANDMARKS)
            x = np.zeros((self.window_length, num_landmarks, 3), dtype=np.float32)

            for t, frame in enumerate(window):
                for i, lm in enumerate(INPUT_LANDMARKS):
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
            y[start_frame:end_frame] = OUTPUT_GESTURES.index(label)

        # output shape: (total_frames,)
        return y

    def create_set(self, sequences_folder, ann_file, out_file, max_sequence_id=None):
        total_X = np.array([], dtype=np.float32).reshape(
            0, self.window_length, len(INPUT_LANDMARKS), 3
        )
        total_y = np.array([], dtype=np.long)
        total_seq_ids = np.array([], dtype=np.int32)  # track sequence IDs

        annotations = parse_shrec_annotations_file(ann_file, OUTPUT_GESTURES)
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

            total_X = np.concatenate((total_X, X), axis=0)
            total_y = np.concatenate((total_y, y), axis=0)
            total_seq_ids = np.concatenate((total_seq_ids, seq_ids), axis=0)

        GTCNDataset.check_shape(total_X, total_y, self.window_length)
        print(f"X.shape: {total_X.shape}, y.shape: {total_y.shape}")
        GTCNDataset.print_label_distribution(total_y)

        with open(out_file, "wb") as f:
            pickle.dump(
                {"X": total_X, "y": total_y, "seq_ids": total_seq_ids},
                f,
            )
        file_size = os.path.getsize(out_file) / (1024 * 1024)
        print(f"Saved to {out_file} ({file_size:.2f} MB)")


def create_datasets(builder: GTCNDatasetBuilder, suffix: str = ""):
    print("[training set]")
    builder.create_set(
        sequences_folder=SHREC_TRAINING_DATASET_FOLDER + "sequences/",
        ann_file=SHREC_TRAINING_DATASET_FOLDER + "annotations_revised_training.txt",
        out_file=DEFAULT_DATASET_FOLDER + f"training_{suffix}.pkl",
    )

    print("\n[test set]")
    builder.create_set(
        sequences_folder=SHREC_TEST_DATASET_FOLDER + "sequences/",
        ann_file=SHREC_TEST_DATASET_FOLDER + "annotations_revised.txt",
        out_file=DEFAULT_DATASET_FOLDER + f"test_{suffix}.pkl",
    )
