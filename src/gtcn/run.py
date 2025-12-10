from src.gtcn.train import train_final_model
from src.gtcn.optimize import find_optimize_params

# best_params = find_optimize_params(n_trials=50)


best_params = {
    'GCN_HIDDEN_DIM': 16,
    'GCN_DROPOUT': 0.37599172332941766,
    'TCN_HIDDEN_DIM': 128,
    'TCN_KERNEL_SIZE': 5,
    'TCN_DILATIONS': (1, 2, 4, 8, 16),
    'TCN_DROPOUT': 0.32371265162044416, 
    'CLASS_HIDDEN_DIM': 32,
    'learning_rate': 0.001450345570851721,
    'weight_decay': 5.5313835406154854e-05
}
train_final_model(best_params, epochs=200)

# nohup python -u -m src.gtcn.run &