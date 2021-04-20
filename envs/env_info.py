from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel

def get_env_name(game_name):
    if game_name == "PredatorPrey":
        return 'PredatorPrey_Game/PredatorPrey'

def get_env_info(env_name, args):
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name="envs/{0}".format(env_name) if args.run_unity_editor is not True else None,
                           #    no_graphics=True,
                           side_channels=[engine_configuration_channel])

    env.reset()

    behavior_name_list = list(env.behavior_specs.keys())

    dec, _ = env.get_steps(behavior_name_list[0]) #Coordinator
    state = dec.obs[0]

    dec, _ = env.get_steps(behavior_name_list[1])  # Agent
    obs = dec.obs[0]

    env_arg = {
        "n_agents": dec.action_mask[0].shape[0],
        "n_actions": dec.action_mask[0].shape[1],
        "state_shape": state.shape[1],
        "obs_shape": obs.shape[1],
        "episode_limit": 160
    }

    return env, env_arg, engine_configuration_channel
