from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from mlagents_envs.environment import ActionTuple


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

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def set_learner(self, learner):
        return

    def run(self, test_mode=False):
        self.reset()

        behavior_name = list(self.env.behavior_specs.keys())[0]

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            state, obs, avail_actions = self.get_env_info_unity(behavior_name)

            pre_transition_data = {
                "state": [state],
                "avail_actions": [avail_actions],
                "obs": [obs]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            detached_action = actions.detach().numpy()
            action_tuple = ActionTuple()
            action_tuple.add_discrete(detached_action)
            self.env.set_actions(behavior_name, action_tuple)
            self.env.step()

            dec, term = self.env.get_steps(behavior_name)
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

        state, obs, avail_actions = self.get_env_info_unity(behavior_name)

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
        #cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
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

    def get_env_info_unity(self, behavior_name):
        dec, term = self.env.get_steps(behavior_name)
        avail_actions = np.array(dec.action_mask).squeeze(axis=1)
        avail_actions_float = np.zeros_like(avail_actions, dtype=np.float)
        avail_actions_float[avail_actions == False] = 1
        avail_actions = avail_actions_float
        data = dec.obs[0]

        state = data[0, :self.args.state_shape]
        obs = data[0, self.args.state_shape:].reshape(self.args.n_agents, -1)

        return state, obs, avail_actions
