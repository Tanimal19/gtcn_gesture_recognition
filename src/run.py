# nohup python -u -m src.run &

from src import DEVICE
from src.gtcn.test import test_model as test_gtcn_model
from src.mhead.test import test_model as test_mhead_model
from src.pthresh.test import test_model as test_pthresh_model


test_set_path = "./src/gtcn/datasets/test.pkl"
gtcn_model_path = "./src/gtcn/models/best_model.pth"
mhead_model_path = "./src/mhead/models/best_model.pth"
pthresh_model_path = "./src/pthresh/models/best_model.pth"


print(f"using device: {DEVICE}")


# test GTCN model
try:
    test_gtcn_model(
        model_path=gtcn_model_path,
        test_set_path=test_set_path,
        batch_size=32,
    )
except Exception as e:
    print(f"Error during GTCN model testing: {e}")

# test MHead model
try:
    test_mhead_model(
        model_path=mhead_model_path,
        test_set_path=test_set_path,
        batch_size=32,
    )
except Exception as e:
    print(f"Error during MHead model testing: {e}")

# test PThresh model
try:
    test_pthresh_model(
        model_path=pthresh_model_path,
        test_set_path=test_set_path,
        threshold=0.66,
        batch_size=32,
    )
except Exception as e:
    print(f"Error during PThresh model testing: {e}")


print("\nscript completed.")
