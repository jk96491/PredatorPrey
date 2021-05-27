from run import runing
import config_util as cu
from utils.logging import get_logger


if __name__ == "__main__":
    algorithm = 'RNN_AGENT/qmix'
    #algorithm = 'Role_Learning_Agent/rode'
    game_name = "PredatorPrey"

    config = cu.config_copy(cu.get_config(algorithm))

    logger = get_logger()
    runing(config, logger, game_name)