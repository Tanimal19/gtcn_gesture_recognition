from enum import Enum, auto
from dataclasses import dataclass


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
class DatasetInput:
    sequence_id: str
    frame_index: int
    landmarks: dict[HandLandmark, tuple[float, float, float]]  # x, y, z coordinates


@dataclass
class DatasetOutput:
    sequence_id: str
    gestures: list[
        tuple[GestureLabel, int, int]
    ]  # (gesture_label, start_frame_index, end_frame_index)
