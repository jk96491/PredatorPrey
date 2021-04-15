map_param_registry = {
    "PredatorPrey_Game/PredatorPrey": {
        "n_agents": 3,
        "n_actions": 5,
        "episode_limit": 160,
        "state_shape": 45,
        "obs_shape": 45
    },
}

def get_env_info(env_name):
    arg = map_param_registry[env_name]
    return arg

def get_env_name(game_name):
    if game_name == "PredatorPrey":
        return 'PredatorPrey_Game/PredatorPrey'