import numpy as np
from learners import REGISTRY as le_REGISTRY
import os
import time
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel
from types import SimpleNamespace as SN
from components.transforms import OneHot
from components.episode_buffer import ReplayBuffer
from controller.basic_controller import BasicMAC
from runners.episode_runner import EpisodeRunner
from utils.logging import Logger
from utils.timehelper import time_left, time_str
import datetime
import torch

load_model = False
train_mode = True

run_step = 50000 if train_mode else 0
test_step = 10000

env_name = 'PredatorPrey_Game/PredatorPrey'


def runing(config, _log):
    _config = args_sanity_check(config, _log)
    args = SN(**config)
    args.device = "cuda" if args.use_cuda else "cpu"

    logger = Logger(_log)
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token

    run_sequential(args, logger)


def run_sequential(args, logger):
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name,
                           #    no_graphics=True,
                           side_channels=[engine_configuration_channel])


    args.n_agents = 3
    args.n_actions = 5
    args.state_shape = 45
    args.obs_shape = 45

    episode_limit = 160

    scheme = {
        "state": {"vshape": args.state_shape},
        "obs": {"vshape": args.obs_shape, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, episode_limit + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    mac = BasicMAC(buffer.scheme, groups, args)

    runner = EpisodeRunner(args=args, logger=logger, env=env)

    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # 유니티 환경 경로 설정 (file_name)

    env.reset()
    # 유니티 브레인 설정
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0)

    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")

def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not torch.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
