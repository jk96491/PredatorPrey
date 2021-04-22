from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from mlagents_envs.environment import ActionTuple
from utils.env_utils import get_env_info

# 이 부분은 유니티 환경과 파이썬의 소통을 통한 환경 수행을 담당합니다.
class EpisodeRunner:

    def __init__(self, args, logger, env):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env
        self.episode_limit = self.args.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def close_env(self):
        self.env.close()

    # 환경을 reset 합니다.
    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    # 환경을 수행 합니다.
    def run(self, test_mode=False):
        self.reset()

        behavior_name_list = list(self.env.behavior_specs.keys())
        Coordinator = behavior_name_list[0]

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)    # agent의 history(trajectory) 정보를 초기화 합니다.

        while not terminated:
            state, obs, avail_actions = get_env_info(behavior_name_list, self.env, self.args.n_agents)

            pre_transition_data = {
                "state": [state],
                "avail_actions": [avail_actions],
                "obs": [obs]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # 지금까지 발생한 정보들을 agent에게 전달합니다.
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            detached_action = actions.detach().numpy()
            action_tuple = ActionTuple()
            action_tuple.add_discrete(detached_action)
            self.env.set_actions(Coordinator, action_tuple)
            self.env.step()

            dec, term = self.env.get_steps(Coordinator)
            terminated = len(term.agent_id) > 0
            reward = term.reward if terminated else dec.reward

            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != terminated,)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        state, obs, avail_actions = get_env_info(behavior_name_list, self.env, self.args.n_agents)

        last_data = {
            "state": [state],
            "avail_actions": [avail_actions],
            "obs": [obs]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

