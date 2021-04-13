import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel

load_model = False
train_mode = True

run_step = 50000 if train_mode else 0
test_step = 10000

env_name = 'PredatorPrey_Game/PredatorPrey'

if __name__ == '__main__':
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
    #agent = A2CAgent()

    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            #if train_mode:
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

