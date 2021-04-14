import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel
from types import SimpleNamespace as SN
from components.transforms import OneHot
from components.episode_buffer import ReplayBuffer
from controller.basic_controller import BasicMAC
from runners.episode_runner import EpisodeRunner
from utils.logging import Logger
import torch

load_model = False
train_mode = True

run_step = 50000 if train_mode else 0
test_step = 10000

env_name = 'PredatorPrey_Game/PredatorPrey'


def runing(config, _log):
    _config = args_sanity_check(config, _log)
    args = SN(**config)

    logger = Logger(_log)

    run_sequential(args, logger)


def run_sequential(args, logger):
    args.n_agents = 3
    args.n_actions = 5
    args.state_shape = 24
    args.obs_shape = 120

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

    runner = EpisodeRunner(args=args, logger=logger)

    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # 유니티 환경 경로 설정 (file_name)
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=None,
                           #    no_graphics=True,
                           side_channels=[engine_configuration_channel])
    env.reset()
    # 유니티 브레인 설정
    group_name = list(env.behavior_specs.keys())[0]
    group_spec = env.behavior_specs[group_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
    dec, term = env.get_steps(group_name)

    # A2CAgent 클래스를 agent로 정의
    # agent = A2CAgent()

    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            # if train_mode:
            #    agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

        data = dec.obs[0]

        state = data[0, :24]
        obs = data[0, 24:]

        action = np.array([[2, 3, 1]])
        action_tuple = ActionTuple()
        action_tuple.add_discrete(action)
        env.set_actions(group_name, action_tuple)
        env.step()

        dec, term = env.get_steps(group_name)

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
