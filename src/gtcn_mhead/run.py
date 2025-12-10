from src.gtcn_mhead.create_training_set import create_training_set
from src.gtcn_mhead.train import train_model
from src.dataset_utils import SHREC_TRAINING_DATASET_FOLDER, SHREC_TEST_DATASET_FOLDER

# create training sets
training_set_path = "./src/gtcn_mhead/datasets/train.pkl"
create_training_set(
    sequences_folder=SHREC_TRAINING_DATASET_FOLDER + "sequences/",
    ann_file=SHREC_TRAINING_DATASET_FOLDER + "annotations_revised_training.txt",
    out_file=training_set_path,
)

test_set_path = "./src/gtcn_mhead/datasets/test.pkl"
# create_training_set(
#     sequences_folder=SHREC_TEST_DATASET_FOLDER + "sequences/",
#     ann_file=SHREC_TEST_DATASET_FOLDER + "annotations_revised.txt",
#     shift=5,
#     out_file=test_set_path,
# )

# nohup python -u -m src.gtcn.run &
