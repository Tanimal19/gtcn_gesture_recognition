from src.utils import GestureLabel, HandLandmark

INPUT_LANDMARKS = [
    HandLandmark.thumbA,
    HandLandmark.thumbB,
    HandLandmark.thumbEnd,
    HandLandmark.indexA,
    HandLandmark.indexB,
    HandLandmark.indexC,
    HandLandmark.indexEnd,
    HandLandmark.middleA,
    HandLandmark.middleB,
    HandLandmark.middleC,
    HandLandmark.middleEnd,
]
INPUT_DIMENSIONS = 3  # x, y, z
OUTPUT_GESTURES = [
    GestureLabel.NONE,
    GestureLabel.GRAB,
    GestureLabel.PINCH,
    GestureLabel.TAP,
    GestureLabel.DENY,
    GestureLabel.KNOB,
    GestureLabel.EXPAND,
]
