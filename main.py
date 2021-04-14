from run import runing
import config_util as cu
import numpy as np
import torch
import random

algorithm = 'RNN_AGENT/qmix'
config = cu.config_copy(cu.get_config(algorithm))

random_Seed = random.randrange(0, 16546)

np.random.seed(random_Seed)
torch.manual_seed(random_Seed)

if __name__ == "__main__":
    runing(config)