import os
import time
from os.path import dirname, abspath
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controller import REGISTRY as mac_REGISTRY
from types import SimpleNamespace as SN
from components.transforms import OneHot
from components.episode_buffer import ReplayBuffer
from envs.env_info import get_env_info
from envs.env_info import get_env_name
from utils.logging import Logger
from utils.timehelper import time_left, time_str
import datetime
import torch


def runing(config, _log, game_name):

    # config 파일로 부터 args 정보를 로드 합니다.
    _config = args_sanity_check(config, _log)
    args = SN(**config)
    args.device = "cuda" if args.use_cuda else "cpu"

    env_name = get_env_name(game_name)

    # log 기능을 활성화 합니다.
    logger = Logger(_log)
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token

    # 텐서보드 기능을 준비 합니다.
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs/{}".format(game_name))
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # 실험을 시작 합니다.
    run_sequential(args, logger, env_name)


def run_sequential(args, logger, env_name):
    # 환경의 여러가지 정보들을 가져 옵니다.
    env, env_arg, engine_configuration_channel = get_env_info(env_name, args)

    # 가져온 환경 정보를 args에 세팅 합니다.
    args.n_agents = env_arg["n_agents"]
    args.n_actions = env_arg["n_actions"]
    args.state_shape = env_arg["state_shape"]
    args.obs_shape = env_arg["obs_shape"]
    args.episode_limit = env_arg["episode_limit"]

    runner = r_REGISTRY[args.runner](args=args, logger=logger, env=env)

    # 환경에서 발생한 정보를 저장하기 위한 ReplayBuffer를 초기화 합니다.
    scheme, groups, preprocess = get_data_infos(args)
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, args.episode_limit + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # agent 및 coordinator 모두 초기화 합니다.
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if is_set_checkpoint(args, logger, learner, runner) is True:
        return

    # 학습을 수행 합니다.
    train(args, logger,learner, runner, buffer, engine_configuration_channel)


def train(args, logger, learner, runner, buffer, engine_configuration_channel):
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        engine_configuration_channel.set_configuration_parameters(time_scale=args.learning_time_scale)

        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            # coordinator를 학습 합니다.
            learner.train(episode_sample, runner.t_env, episode)

        n_test_runs = max(1, args.test_nepisode // runner.batch_size)

        # 일정 주기로 Test를 진행 합니다.
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            engine_configuration_channel.set_configuration_parameters(time_scale=args.test_time_scale)
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        # 일정 주기로 학습된 가중치를 저장 합니다.
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


# 환경의 여러가지 정보들을 지정 합니다.
def get_data_infos(args):
    scheme = {
        "state": {"vshape": args.state_shape},
        "obs": {"vshape": args.obs_shape, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.int},
        "role_avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
        "roles": {"vshape": (1,), "group": "agents", "dtype": torch.long},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    return scheme, groups, preprocess


# 학습된 가중치를 사용 합니다.
def is_set_checkpoint(args, logger, learner, runner):
    just_testing = False

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate :
            evaluate_sequential(args, runner)
            just_testing = True

        return just_testing


# 안정성 검사를 수행합니다.
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


# 학습된 가중치의 평가를 진행 합니다.
def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    runner.close_env()
