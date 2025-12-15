# nohup python -u -m src.run &

from src.dataset_builder import (
    GTCNDatasetBuilder,
    create_datasets,
    DEFAULT_DATASET_FOLDER,
)
from src.gtcn.model import GTCNParams
from src.train import GTCNTrainParams, train_model, DEFAULT_MODEL_FOLDER
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
)
PROBTHRESHOLD_CLASSIFIER_ARCHITECTURE = GTCNParams(
    id="probthreshold_classifier_model",
    GCN_CLASS=GCNLayerFingerPool,
    TCN_CLASS=TCNLayerLastStep,
    CLASSIFIER_CLASS=ProbThresholdClassifier,
)


# create datasets with different window length and peek values
datasets = {
    "base": (15, 0),
    "win10": (10, 0),
    "win30": (30, 0),
    "peek5": (15, 5),
}
for suffix, (window_length, peek) in datasets.items():
    print(
        f"=== Creating dataset with window length {window_length} and peek {peek} ==="
    )
    builder = GTCNDatasetBuilder(window_length=window_length, peek=peek)
    create_datasets(builder, suffix)


# train models with different architectures
model_settings = [
    (BASE_ARCHITECTURE, "base", "base.pth"),
    (WIN10_ARCHITECTURE, "win10", "win10.pth"),
    (WIN30_ARCHITECTURE, "win30", "win30.pth"),
    (BASE_ARCHITECTURE, "peek5", "peek5.pth"),
    (GCN_NOPOOL_ARCHITECTURE, "base", "gcn_nopool.pth"),
    (TCN_MEANPOOL_ARCHITECTURE, "base", "tcn_meanpool.pth"),
    (TCN_WEIGHTPOOL_ARCHITECTURE, "base", "tcn_weightpool.pth"),
    (DOUBLEHEAD_CLASSIFIER_ARCHITECTURE, "base", "doublehead_classifier.pth"),
    (PROBTHRESHOLD_CLASSIFIER_ARCHITECTURE, "base", "probthreshold_classifier.pth"),
]
history: dict[str, list[float]] = {}
for architecture, dataset_suffix, model_filename in model_settings:
    print(f"=== Training model '{architecture.id}' on dataset '{dataset_suffix}' ===")
    train_params = GTCNTrainParams(
        model_params=architecture,
        epochs=10,
    )
    history[model_filename] = train_model(
        train_params,
        training_dataset_path=DEFAULT_DATASET_FOLDER + f"training_{dataset_suffix}.pkl",
        model_path=DEFAULT_MODEL_FOLDER + model_filename,
    )

# save training history
with open("training_history.txt", "w") as f:
    for model_filename, loss_history in history.items():
        f.write(f"{model_filename}: {loss_history}\n")
