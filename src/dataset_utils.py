from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict
import os


SHREC_TRAINING_DATASET_FOLDER = "./shrec_2021/training_set/"
SHREC_TEST_DATASET_FOLDER = "./shrec_2021/test_set/"


class GestureLabel(Enum):
    # static gestures
    ONE = auto()
    TWO = auto()
    THREE = auto()
    FOUR = auto()
    OK = auto()
    MENU = auto()
    POINTING = auto()
    # dynamic gestures
    LEFT = auto()
    RIGHT = auto()
    CIRCLE = auto()
    V = auto()
    CROSS = auto()
    GRAB = auto()
    PINCH = auto()
    TAP = auto()
    DENY = auto()
    KNOB = auto()
    EXPAND = auto()


class HandLandmark(Enum):
    palm = 0
    thumbA = 1
    thumbB = 2
    thumbEnd = 3
    indexA = 4
    indexB = 5
    indexC = 6
    indexEnd = 7
    middleA = 8
    middleB = 9
    middleC = 10
    middleEnd = 11
    ringA = 12
    ringB = 13
    ringC = 14
    ringEnd = 15
    pinkyA = 16
    pinkyB = 17
    pinkyC = 18
    pinkyEnd = 19


@dataclass
class DSequenceFrame:
    frame_index: int
    landmarks: Dict[HandLandmark, tuple[float, float, float]]  # (x, y, z)


@dataclass
class DSequence:
    sequence_id: int
    frames: List[DSequenceFrame]


@dataclass
class DAnnotation:
    sequence_id: int
    gestures: List[
        tuple[GestureLabel, int, int]
    ]  # (gesture_label, start_frame_index, end_frame_index)


# --------------------------------------
# Parsing SHREC21 files to dataclasses
# --------------------------------------
VALUES_PER_LANDMARK = 7  # x,y,z,qx,qy,qz,qw


def parse_shrec_sequence_file(filepath: str) -> DSequence:
    sequence_id = os.path.splitext(os.path.basename(filepath))[0]

    frames: List[DSequenceFrame] = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    for frame_idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        values = [float(v) for v in line.split(";") if v]
        assert len(values) == len(HandLandmark) * VALUES_PER_LANDMARK

        landmarks = {}

        for i, lm in enumerate(HandLandmark):
            base = i * VALUES_PER_LANDMARK
            x, y, z = values[base : base + 3]  # ignore quaternion
            landmarks[lm] = (x, y, z)

        frames.append(
            DSequenceFrame(
                frame_index=frame_idx,
                landmarks=landmarks,
            )
        )

    return DSequence(
        sequence_id=int(sequence_id),
        frames=frames,
    )


def parse_shrec_annotations_file(filepath: str) -> List[DAnnotation]:
    results = []

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 2:
                continue

            sequence_id = parts[0]
            gesture_entries = parts[1:-1]  # last entry is empty after last semicolon

            gestures = []
            # process triplets: LABEL, start, end
            for i in range(0, len(gesture_entries), 3):
                label = GestureLabel[gesture_entries[i]]
                start_f = int(gesture_entries[i + 1])
                end_f = int(gesture_entries[i + 2])
                gestures.append((label, start_f, end_f))

            results.append(
                DAnnotation(
                    sequence_id=int(sequence_id),
                    gestures=gestures,
                )
            )

    return results
