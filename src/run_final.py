# run script for optimizing and training the main model
# nohup python -u -m src.run_final &

from src.dataset_builder import (
    GTCNDatasetBuilder,
    create_datasets,
    DEFAULT_DATASET_FOLDER,
)
from src.optimize import run_optimize
from src.train import train_model, DEFAULT_MODEL_FOLDER
from src.test import test_model
from src.gtcn.model import GTCNParams
from src.gtcn.gcn import GCNLayerFingerPool
from src.gtcn.tcn import TCNLayerLastStep
from src.gtcn.classifier import DoubleHeadClassifier


FINAL_ARCHITECTURE = GTCNParams(
    id="final",
    GCN_CLASS=GCNLayerFingerPool,
    TCN_CLASS=TCNLayerLastStep,
    CLASSIFIER_CLASS=DoubleHeadClassifier,
    WINDOW_LENGTH=15,
)


# create final datasets
# builder = GTCNDatasetBuilder(window_length=15, peek=5)
# create_datasets(builder, suffix="final")

# optimize hyperparameters
best_training_params = run_optimize(
    study_name="final_model_optimization",
    n_trials=50,
    training_set_path=DEFAULT_DATASET_FOLDER + "train_final.pkl",
    base_params=FINAL_ARCHITECTURE,
)

# train final model with best hyperparameters
best_training_params.model_path = DEFAULT_MODEL_FOLDER + "final.pth"
best_training_params.epochs = 200
train_model(best_training_params)


# test final model
results = test_model(
    test_dataset_path=DEFAULT_DATASET_FOLDER + "test_final.pkl",
    model_path=DEFAULT_MODEL_FOLDER + "final.pth",
)

import pandas as pd

df_rows = []
for seq_result in results:
    seq_id = seq_result["seq_id"]
    truths = seq_result["truths"]
    predictions = seq_result["predictions"]
    df_rows.append(
        {
            "seq_id": seq_id,
            "truths": truths,
            "predictions": predictions,
        }
    )
df = pd.DataFrame(df_rows)
df.to_csv("final_testing_results.csv", index=False)
