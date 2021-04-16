from run import runing
import config_util as cu
import numpy as np
import torch
import random
from utils.logging import get_logger


if __name__ == "__main__":
    algorithm = 'RNN_AGENT/qmix'
    game_name = "PredatorPrey"
    run_unity_editor = False         # True -> 유니티와 직접 소통됨, False -> 빌드된 exe 파일과 소통됨

    config = cu.config_copy(cu.get_config(algorithm))

    random_Seed = random.randrange(0, 16546)

    np.random.seed(random_Seed)
    torch.manual_seed(random_Seed)

    logger = get_logger()
    runing(config, logger, game_name, run_unity_editor)