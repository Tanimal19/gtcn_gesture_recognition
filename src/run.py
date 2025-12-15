# nohup python -u -m src.run &

from src.dataset_builder import (
    GTCNDatasetBuilder,
    create_datasets,
    DEFAULT_DATASET_FOLDER,
)
from src.gtcn.model import GTCNParams
from src.train import GTCNTrainParams, train_model, DEFAULT_MODEL_FOLDER
from src.test import test_model
from src.gtcn.gcn import GCNLayerFingerPool, GCNLayerNoPool
from src.gtcn.tcn import TCNLayerLastStep, TCNLayerMeanPool, TCNLayerWeightPool
from src.gtcn.classifier import (
    RegularClassifier,
    DoubleHeadClassifier,
    ProbThresholdClassifier,
)


BASE_ARCHITECTURE = GTCNParams(
    id="base_model",
    GCN_CLASS=GCNLayerFingerPool,
    TCN_CLASS=TCNLayerLastStep,
    CLASSIFIER_CLASS=RegularClassifier,
    WINDOW_LENGTH=15,
)

# architectures of different window lengths
WIN10_ARCHITECTURE = GTCNParams(
    id="win10_model",
    GCN_CLASS=GCNLayerFingerPool,
    TCN_CLASS=TCNLayerLastStep,
    CLASSIFIER_CLASS=RegularClassifier,
    WINDOW_LENGTH=10,
)
WIN30_ARCHITECTURE = GTCNParams(
    id="win30_model",
    GCN_CLASS=GCNLayerFingerPool,
    TCN_CLASS=TCNLayerLastStep,
    CLASSIFIER_CLASS=RegularClassifier,
    WINDOW_LENGTH=30,
    TCN_CHANNELS=[64, 64, 64, 64],
    TCN_DILATIONS=[1, 2, 4, 8],  # increase receptive field
)

# architectures of different gcn/tcn/classifier modules
GCN_NOPOOL_ARCHITECTURE = GTCNParams(
    id="gcn_nopool_model",
    GCN_CLASS=GCNLayerNoPool,
    TCN_CLASS=TCNLayerLastStep,
    CLASSIFIER_CLASS=RegularClassifier,
)
TCN_MEANPOOL_ARCHITECTURE = GTCNParams(
    id="tcn_meanpool_model",
    GCN_CLASS=GCNLayerFingerPool,
    TCN_CLASS=TCNLayerMeanPool,
    CLASSIFIER_CLASS=RegularClassifier,
)
TCN_WEIGHTPOOL_ARCHITECTURE = GTCNParams(
    id="tcn_weightpool_model",
    GCN_CLASS=GCNLayerFingerPool,
    TCN_CLASS=TCNLayerWeightPool,
    CLASSIFIER_CLASS=RegularClassifier,
)
DOUBLEHEAD_CLASSIFIER_ARCHITECTURE = GTCNParams(
    id="doublehead_classifier_model",
    GCN_CLASS=GCNLayerFingerPool,
    TCN_CLASS=TCNLayerLastStep,
    CLASSIFIER_CLASS=DoubleHeadClassifier,
    DOUBLE_HEAD_BCE_WEIGHT=0.1,
)
PROBTHRESHOLD_CLASSIFIER_ARCHITECTURE = GTCNParams(
    id="probthreshold_classifier_model",
    GCN_CLASS=GCNLayerFingerPool,
    TCN_CLASS=TCNLayerLastStep,
    CLASSIFIER_CLASS=ProbThresholdClassifier,
    PROB_THRESHOLD=0.6,
)


def generate_datasets():
    datasets = {
        "base": (15, 0),
        "win10": (10, 0),
        "win30": (30, 0),
        "peek5": (15, 5),
    }
    for suffix, (window_length, peek) in datasets.items():
        print(
            f"\n=== Creating dataset with window length {window_length} and peek {peek} ==="
        )
        builder = GTCNDatasetBuilder(window_length=window_length, peek=peek)
        create_datasets(builder, suffix)


def train_models():
    history: dict[str, list[float]] = {}

    def start_train(
        model_name: str,
        architecture: GTCNParams,
        dataset_suffix: str,
        weight_mode: str | None = None,
    ):
        params = GTCNTrainParams(
            model_params=architecture,
            training_dataset_path=DEFAULT_DATASET_FOLDER
            + f"training_{dataset_suffix}.pkl",
            model_path=DEFAULT_MODEL_FOLDER + f"{model_name}.pth",
            weight_mode=weight_mode,
            epochs=150,
        )
        print(f"\n=== Training model '{model_name}' on dataset '{dataset_suffix}' ===")
        history[model_name] = train_model(params)

    start_train("base", BASE_ARCHITECTURE, "base", None)
    start_train("base_simple", BASE_ARCHITECTURE, "base", "simple")
    start_train("base_balanced", BASE_ARCHITECTURE, "base", "balanced")
    start_train("base_peek5", BASE_ARCHITECTURE, "peek5", None)
    start_train("win10", WIN10_ARCHITECTURE, "win10", None)
    start_train("win30", WIN30_ARCHITECTURE, "win30", None)
    start_train("gcn_nopool", GCN_NOPOOL_ARCHITECTURE, "base", None)
    start_train("tcn_meanpool", TCN_MEANPOOL_ARCHITECTURE, "base", None)
    start_train("tcn_weightpool", TCN_WEIGHTPOOL_ARCHITECTURE, "base", None)
    start_train(
        "doublehead_classifier", DOUBLEHEAD_CLASSIFIER_ARCHITECTURE, "base", None
    )
    start_train(
        "probthreshold_classifier", PROBTHRESHOLD_CLASSIFIER_ARCHITECTURE, "base", None
    )

    # save training history
    with open("training_history.txt", "w") as f:
        for model_filename, loss_history in history.items():
            f.write(f"{model_filename}: {loss_history}\n")


def test_models():
    tests = [  # (model_name, dataset_suffix)
        ("base", "base"),
        ("base_simple", "base"),
        ("base_balanced", "base"),
        ("base_peek5", "peek5"),
        ("win10", "win10"),
        ("win30", "win30"),
        ("gcn_nopool", "base"),
        ("tcn_meanpool", "base"),
        ("tcn_weightpool", "base"),
        ("doublehead_classifier", "base"),
        ("probthreshold_classifier", "base"),
    ]

    results = {}
    for model_name, dataset_suffix in tests:
        print(f"\n=== Testing model '{model_name}' on dataset '{dataset_suffix}' ===")
        results[model_name] = test_model(
            test_dataset_path=DEFAULT_DATASET_FOLDER + f"testing_{dataset_suffix}.pkl",
            model_path=DEFAULT_MODEL_FOLDER + f"{model_name}.pth",
        )

    # save testing results
    with open("testing_results.txt", "w") as f:
        for model_name, result in results.items():
            f.write(f"{model_name}: {result}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GTCN Running Script")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate datasets",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train models",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test models",
    )
    args = parser.parse_args()

    if args.generate:
        generate_datasets()
    if args.train:
        train_models()
    if args.test:
        test_models()
