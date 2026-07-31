import random
import numpy as np
import torch

SEED = 45

# Result of Hyperparameter Search
LOCKED_CONFIG = {
    "window_size": 10,
    "stride": 5,
    "learning_rate": 0.003,
    "weight_decay": 0.001,
    "batch_size": 16,
    "num_epochs": 25,
    "threshold": 0.5
}

# Needs to be set at the beginning to ensure all testing is reproducible.
def set_seed(seed=45):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)