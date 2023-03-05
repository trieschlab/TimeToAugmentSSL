import time

import numpy as np
from gym import Wrapper


class TimeLimit(Wrapper):
    def __init__(self, env, max_episode_seconds=None, max_episode_steps=None,reset=True):
        super(TimeLimit, self).__init__(env)
        self._max_episode_seconds = max_episode_seconds
        self._max_episode_steps = max_episode_steps
        self.need_reset=reset
        self._elapsed_steps = 0
        self._episode_started_at = None

    @property
    def _elapsed_seconds(self):
        return time.time() - self._episode_started_at

    def _past_limit(self):
        if self._max_episode_steps == -1:
            return False

        """Return true if we are past our limit"""
        if self._max_episode_steps is not None and self._max_episode_steps <= self._elapsed_steps:
            return True

        if self._max_episode_seconds is not None and self._max_episode_seconds <= self._elapsed_seconds:
            return True

        return False

    def step(self, action):
        assert self._episode_started_at is not None, "Cannot call env.step() before calling reset()"

        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        info["past"] = False
        info["over"] = False
        info["true_obs"] = None

        if done:
            info["over"]=True
        if self._past_limit():# or done:
            info["past"]=True

        #info["state"] = observation["state"]
        if self.need_reset and (self._past_limit() or done):
            if isinstance(observation,dict):
                info["true_obs"] = observation["observation"]
            else:
                info["true_obs"] = np.copy(observation)
            observation = self.reset()
            done=True
        # observation["state"update_ne] = observation["observation"]
        return observation, reward, done, info

    def reset(self):
        self._episode_started_at = time.time()
        self._elapsed_steps = 0
        obs = self.env.reset()
        # obs["state"]=obs["observation"]
        return obs

