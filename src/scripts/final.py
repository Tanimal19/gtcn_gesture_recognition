from src.gtcn.model import GTCNParams
from src.gtcn.gcn import GCNLayerFingerPool
from src.gtcn.tcn import TCNLayerMeanPool
from src.gtcn.classifier import RegularClassifier, DoubleHeadClassifier, ProbThresholdClassifier


FINAL_ARCHITECTURE = GTCNParams(
    id="final",
    GCN_CLASS=GCNLayerFingerPool,
    GCN_DIMS=[16, 32],
    GCN_DROPOUT=0.2,
    TCN_CLASS=TCNLayerMeanPool,
    TCN_CHANNELS=[64, 64, 64, 64],
    TCN_KERNEL_SIZE=3,
    TCN_DILATIONS=[1, 2, 4, 8],
    TCN_DROPOUT=0.3,
    CLASSIFIER_CLASS=DoubleHeadClassifier,
    DOUBLE_HEAD_BCE_WEIGHT = 0.1,
    PROB_THRESHOLD = 0.6,
    WINDOW_LENGTH=15,
)


# train final model
from src.dataset_builder import DEFAULT_DATASET_FOLDER
from src.train import train_model, DEFAULT_MODEL_FOLDER, GTCNTrainParams

training_settings = [
    ("final", "base"),
]

loss_history = {}
for model_name, dataset_suffix in training_settings:
    training_params = GTCNTrainParams(
        model_params=FINAL_ARCHITECTURE,
        training_dataset_path=DEFAULT_DATASET_FOLDER + f"train_{dataset_suffix}.pkl",
        model_path=DEFAULT_MODEL_FOLDER + f"{model_name}.pth",
        weight_mode="simple",
        learning_rate=1e-3,
        epochs=200,
    )
    loss_history[model_name] = train_model(training_params)


# # test final model
from src.test import test_model
import pandas as pd

results = {}
for model_name, dataset_suffix in training_settings:
    results[model_name] = test_model(
        test_dataset_path=DEFAULT_DATASET_FOLDER + f"test_{dataset_suffix}.pkl",
        model_path=DEFAULT_MODEL_FOLDER + f"{model_name}.pth",
    )

df_rows = []
for model_name, test_result in results.items():
    for seq_result in test_result:
        seq_id = seq_result["seq_id"]
        truths = seq_result["truths"]
        predictions = seq_result["predictions"]
        df_rows.append(
            {
                "model": model_name,
                "seq_id": seq_id,
                "truths": truths,
                "predictions": predictions,
            }
        )
df = pd.DataFrame(df_rows)
df.to_csv("final_testing_results.csv", index=False)


# evaluate final model results
from src.evaluate import evaluate_csv

evaluate_csv(csv_path="final_testing_results.csv")
