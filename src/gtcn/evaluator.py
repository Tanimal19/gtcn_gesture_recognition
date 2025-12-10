import torch
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
from src.gtcn.model import GTCNModel, GTCNHyperParams
from src.dataset_utils import (
    DSequence,
    DAnnotation,
    GestureLabel,
    HandLandmark,
    parse_shrec_sequence_file,
    parse_shrec_annotations_file,
    SHREC_TEST_DATASET_FOLDER,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path: str) -> GTCNModel:
    """Load trained GTCN model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Reconstruct hyperparameters
    hyperparams_dict = checkpoint["hyperparams"]
    hyperparams = GTCNHyperParams(
        id=hyperparams_dict.get("id", "model"),
        GCN_HIDDEN_DIM=hyperparams_dict["GCN_HIDDEN_DIM"],
        GCN_DROPOUT=hyperparams_dict["GCN_DROPOUT"],
        TCN_HIDDEN_DIM=hyperparams_dict["TCN_HIDDEN_DIM"],
        TCN_KERNEL_SIZE=hyperparams_dict["TCN_KERNEL_SIZE"],
        TCN_DILATIONS=tuple(hyperparams_dict["TCN_DILATIONS"]),
        TCN_DROPOUT=hyperparams_dict["TCN_DROPOUT"],
        CLASS_HIDDEN_DIM=hyperparams_dict["CLASS_HIDDEN_DIM"],
    )

    # Load model
    model = GTCNModel(hyperparams).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {model_path}")
    print(
        f"Trained for {checkpoint['epoch']} epochs with loss: {checkpoint['loss']:.4f}"
    )

    return model


def prepare_sequence_windows(sequence: DSequence) -> Tuple[np.ndarray, List[int]]:
    """
    Convert sequence to sliding windows for model input.

    Returns:
        windows: (num_windows, window_length, num_landmarks, 3)
        frame_indices: center frame index for each window
    """
    window_length = GTCNModel.WINDOW_LENGTH
    landmarks_to_use = GTCNModel.LANDMARKS

    # Extract landmark data for entire sequence
    num_frames = len(sequence.frames)
    num_landmarks = len(landmarks_to_use)

    sequence_data = np.zeros((num_frames, num_landmarks, 3), dtype=np.float32)

    for frame_idx, frame in enumerate(sequence.frames):
        for lm_idx, landmark in enumerate(landmarks_to_use):
            x, y, z = frame.landmarks[landmark]
            sequence_data[frame_idx, lm_idx, :] = [x, y, z]

    # Create sliding windows with stride=1
    windows = []
    frame_indices = []

    for i in range(num_frames - window_length + 1):
        window = sequence_data[i : i + window_length]
        windows.append(window)
        # Center frame of the window
        center_frame = i + window_length // 2
        frame_indices.append(center_frame)

    if len(windows) == 0:
        return np.array([]), []

    return np.array(windows), frame_indices


def predict_sequence(
    model: GTCNModel, sequence: DSequence, batch_size: int = 64
) -> Dict[int, GestureLabel]:
    """
    Predict gesture labels for each frame in a sequence.

    Returns:
        Dict mapping frame_index to predicted GestureLabel
    """
    windows, frame_indices = prepare_sequence_windows(sequence)

    if len(windows) == 0:
        print(
            f"Warning: Sequence {sequence.sequence_id} has fewer frames than window length"
        )
        return {}

    # Convert to tensor
    X = torch.tensor(windows, dtype=torch.float32).to(DEVICE)

    # Predict in batches
    predictions = []
    model.eval()

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)

    # Map predictions to frame indices
    frame_predictions = {}
    for frame_idx, pred_idx in zip(frame_indices, predictions):
        gesture = GTCNModel.GESTURES[pred_idx]
        frame_predictions[frame_idx] = gesture

    return frame_predictions


def merge_consecutive_gestures(
    frame_predictions: Dict[int, GestureLabel],
) -> List[Tuple[GestureLabel, int, int]]:
    """
    Merge consecutive frames with same gesture label into gesture segments.

    Returns:
        List of (gesture_label, start_frame, end_frame) tuples
    """
    if not frame_predictions:
        return []

    # Sort by frame index
    sorted_frames = sorted(frame_predictions.items())

    gestures = []
    current_gesture = None
    start_frame = None
    end_frame = None

    for frame_idx, gesture in sorted_frames:
        # Skip NONE gestures
        if gesture == GestureLabel.NONE:
            if current_gesture is not None and current_gesture != GestureLabel.NONE:
                gestures.append((current_gesture, start_frame, end_frame))
                current_gesture = None
            continue

        # Start new gesture or continue current
        if current_gesture is None or current_gesture != gesture:
            # Save previous gesture
            if current_gesture is not None and current_gesture != GestureLabel.NONE:
                gestures.append((current_gesture, start_frame, end_frame))
            # Start new gesture
            current_gesture = gesture
            start_frame = frame_idx
            end_frame = frame_idx
        else:
            # Continue current gesture
            end_frame = frame_idx

    # Add final gesture
    if current_gesture is not None and current_gesture != GestureLabel.NONE:
        gestures.append((current_gesture, start_frame, end_frame))

    return gestures


def filter_short_gestures(
    gestures: List[Tuple[GestureLabel, int, int]], min_length: int = 5
) -> List[Tuple[GestureLabel, int, int]]:
    """Filter out gesture segments that are too short."""
    return [(g, s, e) for g, s, e in gestures if (e - s + 1) >= min_length]


def generate_annotations_for_sequence(
    model: GTCNModel,
    sequence: DSequence,
    min_gesture_length: int = 5,
) -> DAnnotation:
    """Generate annotation for a single sequence."""
    # Get frame-level predictions
    frame_predictions = predict_sequence(model, sequence)

    # Merge into gesture segments
    gestures = merge_consecutive_gestures(frame_predictions)

    # Filter short gestures
    gestures = filter_short_gestures(gestures, min_length=min_gesture_length)

    return DAnnotation(
        sequence_id=sequence.sequence_id,
        gestures=gestures,
    )


def evaluate_and_generate_annotations(
    model_path: str,
    test_folder: str = SHREC_TEST_DATASET_FOLDER,
    output_file: Optional[str] = None,
    min_gesture_length: int = 5,
) -> List[DAnnotation]:
    """
    Evaluate GTCN model on test sequences and generate annotations.

    Args:
        model_path: Path to trained model checkpoint
        test_folder: Folder containing test sequences
        output_file: Optional path to save generated annotations
        min_gesture_length: Minimum number of frames for a valid gesture

    Returns:
        List of DAnnotation objects for all test sequences
    """
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)

    # Get all test sequences
    sequences_folder = os.path.join(test_folder, "sequences")
    sequence_files = sorted(
        [f for f in os.listdir(sequences_folder) if f.endswith(".txt")]
    )

    print(f"\nProcessing {len(sequence_files)} test sequences...")

    annotations = []

    for seq_file in sequence_files:
        seq_path = os.path.join(sequences_folder, seq_file)
        sequence = parse_shrec_sequence_file(seq_path)

        # Generate annotation
        annotation = generate_annotations_for_sequence(
            model, sequence, min_gesture_length
        )
        annotations.append(annotation)

        print(
            f"Sequence {sequence.sequence_id}: {len(annotation.gestures)} gestures detected"
        )

    # Save annotations to file if specified
    if output_file:
        save_annotations(annotations, output_file)
        print(f"\nAnnotations saved to {output_file}")

    return annotations


def save_annotations(annotations: List[DAnnotation], output_file: str):
    """Save annotations in SHREC format."""
    with open(output_file, "w") as f:
        for annotation in annotations:
            line = f"{annotation.sequence_id};"
            for gesture, start, end in annotation.gestures:
                line += f"{gesture.name};{start};{end};"
            f.write(line + "\n")


def compare_with_ground_truth(
    generated_annotations: List[DAnnotation],
    ground_truth_file: str,
) -> Dict:
    """
    Compare generated annotations with ground truth.

    Returns statistics about the predictions.
    """
    ground_truth = parse_shrec_annotations_file(ground_truth_file, GTCNModel.GESTURES)

    # Create lookup dict for ground truth
    gt_dict = {ann.sequence_id: ann for ann in ground_truth}
    gen_dict = {ann.sequence_id: ann for ann in generated_annotations}

    total_gt_gestures = sum(len(ann.gestures) for ann in ground_truth)
    total_gen_gestures = sum(len(ann.gestures) for ann in generated_annotations)

    # Count gesture types
    gt_gesture_counts = {}
    gen_gesture_counts = {}

    for ann in ground_truth:
        for gesture, _, _ in ann.gestures:
            gt_gesture_counts[gesture] = gt_gesture_counts.get(gesture, 0) + 1

    for ann in generated_annotations:
        for gesture, _, _ in ann.gestures:
            gen_gesture_counts[gesture] = gen_gesture_counts.get(gesture, 0) + 1

    stats = {
        "total_sequences": len(ground_truth),
        "total_gt_gestures": total_gt_gestures,
        "total_generated_gestures": total_gen_gestures,
        "gt_gesture_counts": gt_gesture_counts,
        "gen_gesture_counts": gen_gesture_counts,
    }

    return stats


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate GTCN model on test data")
    parser.add_argument(
        "--model",
        type=str,
        default="./src/gtcn/models/best_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--test-folder",
        type=str,
        default=SHREC_TEST_DATASET_FOLDER,
        help="Folder containing test sequences",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/generated_annotations.txt",
        help="Output file for generated annotations",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=5,
        help="Minimum gesture length in frames",
    )
    parser.add_argument(
        "--compare-gt",
        action="store_true",
        help="Compare with ground truth annotations",
    )

    args = parser.parse_args()

    # Generate annotations
    annotations = evaluate_and_generate_annotations(
        model_path=args.model,
        test_folder=args.test_folder,
        output_file=args.output,
        min_gesture_length=args.min_length,
    )

    # Compare with ground truth if requested
    if args.compare_gt:
        gt_file = os.path.join(args.test_folder, "annotations_revised.txt")
        if os.path.exists(gt_file):
            print("\n" + "=" * 50)
            print("Comparison with Ground Truth")
            print("=" * 50)
            stats = compare_with_ground_truth(annotations, gt_file)

            print(f"\nTotal sequences: {stats['total_sequences']}")
            print(f"Ground truth gestures: {stats['total_gt_gestures']}")
            print(f"Generated gestures: {stats['total_generated_gestures']}")

            print("\nGesture distribution:")
            print(f"{'Gesture':<15} {'Ground Truth':<15} {'Generated':<15}")
            print("-" * 45)

            all_gestures = set(stats["gt_gesture_counts"].keys()) | set(
                stats["gen_gesture_counts"].keys()
            )
            for gesture in sorted(all_gestures, key=lambda g: g.value):
                gt_count = stats["gt_gesture_counts"].get(gesture, 0)
                gen_count = stats["gen_gesture_counts"].get(gesture, 0)
                print(f"{gesture.name:<15} {gt_count:<15} {gen_count:<15}")
        else:
            print(f"Ground truth file not found: {gt_file}")

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
