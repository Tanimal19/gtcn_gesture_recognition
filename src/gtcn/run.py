from src.gtcn.create_training_set import create_training_set
from src.gtcn.optimize import find_optimize_params
from src.gtcn.train import train_model
from src.dataset_utils import SHREC_TRAINING_DATASET_FOLDER

# create_training_set(
#     sequences_folder=SHREC_TRAINING_DATASET_FOLDER + "sequences/",
#     ann_file=SHREC_TRAINING_DATASET_FOLDER + "annotations_revised_training.txt",
#     max_sequence_id=None,
# )

# best_params = find_optimize_params(n_trials=20)
example_params = {
    "GCN_HIDDEN_DIM": 16,
    "GCN_DROPOUT": 0.3,
    "TCN_HIDDEN_DIM": 128,
    "TCN_KERNEL_SIZE": 5,
    "TCN_DILATIONS": (1, 2, 4, 8, 16),
    "TCN_DROPOUT": 0.3,
    "CLASS_HIDDEN_DIM": 32,
    "learning_rate": 1.5e-3,
}
train_model(example_params, epochs=100)

# nohup python -u -m src.gtcn.run &
