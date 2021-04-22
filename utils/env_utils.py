import numpy as np

# 현재 시점의 state, obs, avail_actions 정보들을 환경으로부터 가져 옵니다.
def get_env_info(behavior_name_list, env, n_agents):
    Coordinator = behavior_name_list[0]
    Agents = behavior_name_list[1]

    dec, term = env.get_steps(Agents)
    obs = dec.obs[0].reshape(n_agents, -1)

    dec, term = env.get_steps(Coordinator)
    avail_actions = np.array(dec.action_mask).squeeze(axis=1)
    avail_actions_float = np.zeros_like(avail_actions, dtype=np.float)
    avail_actions_float[avail_actions == False] = 1
    avail_actions = avail_actions_float.tolist()
    state = dec.obs[0]

    return state, obs, avail_actions