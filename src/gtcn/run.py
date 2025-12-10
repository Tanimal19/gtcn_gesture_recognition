from src.gtcn.create_training_set import create_training_set
from src.gtcn.optimize import find_optimize_params
from src.gtcn.train import train_model
from src.dataset_utils import SHREC_TRAINING_DATASET_FOLDER

# create training sets
regular_training_set_path = "./src/gtcn/datasets/train.pkl"
create_training_set(
    sequences_folder=SHREC_TRAINING_DATASET_FOLDER + "sequences/",
    ann_file=SHREC_TRAINING_DATASET_FOLDER + "annotations_revised_training.txt",
    out_file=regular_training_set_path,
)

shift5_training_set_path = "./src/gtcn/datasets/train_s5.pkl"
create_training_set(
    sequences_folder=SHREC_TRAINING_DATASET_FOLDER + "sequences/",
    ann_file=SHREC_TRAINING_DATASET_FOLDER + "annotations_revised_training.txt",
    shift=5,
    out_file=shift5_training_set_path,
)

# optimize and train regular training set
regular_best_params = find_optimize_params(
    n_trials=2, training_set_path=regular_training_set_path
)
train_model(params=regular_best_params, training_set_path=regular_training_set_path, model_path="./src/gtcn/datasets/model.pth", epochs=2)

# optimize and train shift=5 training set
shift5_best_params = find_optimize_params(
    n_trials=2, training_set_path=shift5_training_set_path
)
train_model(params=shift5_best_params, training_set_path=shift5_training_set_path, model_path="./src/gtcn/datasets/model_s5.pth", epochs=2)


# nohup python -u -m src.gtcn.run &
