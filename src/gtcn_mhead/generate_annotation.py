import torch
import time
import os
from src.gtcn.model import GTCNModel, GTCNHyperParams
from src.gtcn.train import DEFAULT_MODEL_PATH
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
DEFAULT_ANNOTATION_PATH = "./src/gtcn/datasets/annotations.txt"

DROP_GESTURE_THRESHOLD = 5  # drop gestures shorter than this number of frames
MERGE_FRAMES_THRESHOLD = 10  # if two same gestures are separated by less than this number of frames, merge them


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


def _merge_consecutive_gestures(frame_predictions: dict[int, GestureLabel]):
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


def _apply_thresholds_to_annotation(annotation: DAnnotation) -> DAnnotation:
    # Apply DROP_GESTURE_THRESHOLD: filter out short gestures
    filtered_gestures = []
    for gesture, start, end in annotation.gestures:
        if (end - start + 1) < DROP_GESTURE_THRESHOLD:
            print(
                f"Dropping gesture {gesture.name} at frames {start}:{end} (length {end - start + 1})."
            )
        else:
            filtered_gestures.append((gesture, start, end))

    # Apply MERGE_FRAMES_THRESHOLD: merge same gestures separated by small gaps
    merged_gestures = []
    prev_gesture, prev_start, prev_end = None, None, None
    for i, (gesture, start, end) in enumerate(filtered_gestures):
        if i == 0:
            prev_gesture, prev_start, prev_end = gesture, start, end
        else:
            gap = start - prev_end - 1

            # If same gesture and gap is small enough, merge them
            if gesture == prev_gesture and gap < MERGE_FRAMES_THRESHOLD:
                print(
                    f"Merging gestures {gesture.name} at frames {prev_start}:{prev_end} and {start}:{end}."
                )
                prev_end = end  # extend the end frame
            else:
                merged_gestures.append((prev_gesture, prev_start, prev_end))
                prev_gesture, prev_start, prev_end = gesture, start, end
    if prev_gesture is not None:
        merged_gestures.append((prev_gesture, prev_start, prev_end))

    return DAnnotation(sequence_id=annotation.sequence_id, gestures=merged_gestures)


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
        frame_predictions[frame.frame_index] = GTCNModel.GESTURES[pred]

    raw_gestures = _merge_consecutive_gestures(frame_predictions)

    return DAnnotation(sequence_id=sequence.sequence_id, gestures=raw_gestures)


def generate_annotations(
    model_path: str, annotation_path: str = DEFAULT_ANNOTATION_PATH
):
    model = load_model(model_path)

    sequences_folder = os.path.join(SHREC_TEST_DATASET_FOLDER, "sequences")
    sequence_files = sorted(
        [f for f in os.listdir(sequences_folder) if f.endswith(".txt")]
    )

    print(f"Find {len(sequence_files)} test sequences.")

    raw_annotations = []
    modified_annotations = []
    for seq_file in sequence_files:
        seq_path = os.path.join(sequences_folder, seq_file)
        sequence = parse_shrec_sequence_file(seq_path)

        raw_annotation = generate_annotations_for_sequence(model, sequence)
        raw_annotations.append(raw_annotation)
        modified_annotation = _apply_thresholds_to_annotation(raw_annotation)
        modified_annotations.append(modified_annotation)

    save_annotations(raw_annotations, annotation_path + ".raw")
    save_annotations(modified_annotations, annotation_path)


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
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_ANNOTATION_PATH,
        help="Path to save generated annotations",
    )
    args = parser.parse_args()

    start_time = time.time()
    generate_annotations(model_path=args.model, annotation_path=args.output)
    print(f"Annotation generation completed in {time.time() - start_time} seconds.")


# python -u -m src.gtcn.generate_annotation --model ./src/gtcn/datasets/model.pth --output ./src/gtcn/datasets/annotations.txt
