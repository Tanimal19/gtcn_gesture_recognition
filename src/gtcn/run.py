from src.gtcn.train import train_final_model
from src.gtcn.optimize import find_optimize_params

best_params = find_optimize_params(n_trials=50)
train_final_model(best_params, epochs=100)

# nohup python -u -m src.gtcn.run &