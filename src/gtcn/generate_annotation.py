import torch
import numpy as np
import os
from src.gtcn.model import GTCNModel, GTCNHyperParams
from src.gtcn.train import BEST_MODEL_PATH
from src.gtcn.create_training_set import convert_sequence_to_X
from src.dataset_utils import (
    DSequence,
    DAnnotation,
    GestureLabel,
    parse_shrec_sequence_file,
    SHREC_TEST_DATASET_FOLDER,
)
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANNOTATION_PATH = "./src/gtcn/datasets/generated_annotations.txt"


def load_model(model_path: str) -> GTCNModel:
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Reconstruct hyperparameters
    hyperparams_dict = checkpoint["hyperparams"]
    hyperparams = GTCNHyperParams(
        id=hyperparams_dict.get("id", "model"),
        GCN_HIDDEN_DIM=hyperparams_dict["GCN_HIDDEN_DIM"],
        GCN_DROPOUT=hyperparams_dict["GCN_DROPOUT"],
        TCN_HIDDEN_DIM=hyperparams_dict["TCN_HIDDEN_DIM"],
        TCN_KERNEL_SIZE=hyperparams_dict["TCN_KERNEL_SIZE"],
        TCN_DILATIONS=hyperparams_dict["TCN_DILATIONS"],
        TCN_DROPOUT=hyperparams_dict["TCN_DROPOUT"],
        CLASS_HIDDEN_DIM=hyperparams_dict["CLASS_HIDDEN_DIM"],
    )

    # Load model
    model = GTCNModel(hyperparams).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model Hyperparameters: {hyperparams}")

    return model


def _merge_consecutive_gestures(
    frame_predictions: dict[int, GestureLabel],
) -> list[tuple[GestureLabel, int, int]]:
    if not frame_predictions:
        return []

    gestures = []
    current_gesture = None
    start_frame = None
    end_frame = None

    for frame_idx, gesture in frame_predictions.items():
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


def generate_annotations_for_sequence(
    model: GTCNModel, sequence: DSequence
) -> DAnnotation:
    print(f"Predicting sequence {sequence.sequence_id}.")

    X = convert_sequence_to_X(sequence)
    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    y_predictions = []
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).numpy()
            y_predictions.extend(preds)

    assert len(y_predictions) == len(sequence.frames), "Prediction length mismatch!"

    frame_predictions = {}
    for frame, pred in zip(sequence.frames, y_predictions):
        frame_predictions[frame.frame_index] = GestureLabel(pred)

    gestures = _merge_consecutive_gestures(frame_predictions)

    return DAnnotation(
        sequence_id=sequence.sequence_id,
        gestures=gestures,
    )


def generate_annotations(model_path: str) -> list[DAnnotation]:
    model = load_model(model_path)

    sequences_folder = os.path.join(SHREC_TEST_DATASET_FOLDER, "sequences")
    sequence_files = sorted(
        [f for f in os.listdir(sequences_folder) if f.endswith(".txt")]
    )

    print(f"Find {len(sequence_files)} test sequences.")

    annotations = []
    for seq_file in sequence_files:
        seq_path = os.path.join(sequences_folder, seq_file)
        sequence = parse_shrec_sequence_file(seq_path)

        annotation = generate_annotations_for_sequence(model, sequence)
        annotations.append(annotation)

    save_annotations(annotations, ANNOTATION_PATH)

    return annotations


def save_annotations(annotations: list[DAnnotation], output_file: str):
    with open(output_file, "w") as f:
        for annotation in annotations:
            line = f"{annotation.sequence_id};"
            for gesture, start, end in annotation.gestures:
                line += f"{gesture.name};{start};{end};"
            f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GTCN model on test data")
    parser.add_argument(
        "--model",
        type=str,
        default=BEST_MODEL_PATH,
        help="Path to trained model checkpoint",
    )
    args = parser.parse_args()

    generate_annotations(model_path=args.model)
