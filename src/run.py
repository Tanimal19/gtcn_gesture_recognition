# nohup python -u -m src.run &

import time
from src import DEVICE
from src.dataset_utils import SHREC_TEST_DATASET_FOLDER, SHREC_TRAINING_DATASET_FOLDER
from src.gtcn.dataset import GTCNDatasetBuilder
from src.gtcn.optimize import find_optimize_params as gtcn_find_optimize_params
from src.mhead.optimize import find_optimize_params as mhead_find_optimize_params
from src.gtcn.train import train_model as train_gtcn_model
from src.mhead.train import train_model as train_mhead_model
from src.gtcn.test import test_model as test_gtcn_model
from src.mhead.test import test_model as test_mhead_model


training_set_path = "./src/gtcn/datasets/training.pkl"
test_set_path = "./src/gtcn/datasets/test.pkl"
gtcn_model_path = "./src/gtcn/models/best_model.pth"
mhead_model_path = "./src/mhead/models/best_model.pth"


# generate datasets
# builder = GTCNDatasetBuilder()
# builder.create_set(
#     sequences_folder=SHREC_TRAINING_DATASET_FOLDER + "sequences/",
#     ann_file=SHREC_TRAINING_DATASET_FOLDER + "annotations_revised_training.txt",
#     out_file=training_set_path,
# )
# builder.create_set(
#     sequences_folder=SHREC_TEST_DATASET_FOLDER + "sequences/",
#     ann_file=SHREC_TEST_DATASET_FOLDER + "annotations_revised.txt",
#     out_file=test_set_path,
# )

# ensure files are written before proceeding
time.sleep(1)

print(f"using device: {DEVICE}")

# # optimize GTCN model
# gtcn_best_params = None
# try:
#     gtcn_best_params = gtcn_find_optimize_params(
#         study_name="gtcn_study",
#         training_set_path=training_set_path,
#         n_trials=40,
#     )
# except Exception as e:
#     print(f"Error during GTCN model optimization: {e}")

# optimize MHead model
mhead_best_params = None
try:
    mhead_best_params = mhead_find_optimize_params(
        study_name="mhead_study",
        training_set_path=training_set_path,
        n_trials=40,
    )
except Exception as e:
    print(f"Error during MHead model optimization: {e}")

# # train GTCN model
# try:
#     train_gtcn_model(
#         params=gtcn_best_params,
#         epochs=200,
#         training_set_path=training_set_path,
#         model_path=gtcn_model_path,
#         batch_size=32,
#     )
# except Exception as e:
#     print(f"Error during GTCN model training: {e}")

# train MHead model
try:
    train_mhead_model(
        params=mhead_best_params,
        epochs=200,
        training_set_path=training_set_path,
        model_path=mhead_model_path,
        batch_size=32,
    )
except Exception as e:
    print(f"Error during MHead model training: {e}")

# test GTCN model
# try:
#     test_gtcn_model(
#         model_path=gtcn_model_path,
#         test_set_path=test_set_path,
#         batch_size=32,
#     )
# except Exception as e:
#     print(f"Error during GTCN model testing: {e}")

# test MHead model
try:
    test_mhead_model(
        model_path=mhead_model_path,
        test_set_path=test_set_path,
        batch_size=32,
    )
except Exception as e:
    print(f"Error during MHead model testing: {e}")


print("\nscript completed.")