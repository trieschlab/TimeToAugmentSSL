import time

import gym
import numpy as np
import torch
from gym import Wrapper, ActionWrapper

from models.simple_agents import Nord
from tools.utils import build_default_act_space


class TimeLimitSpe(Wrapper):
    def __init__(self, env, max_episode_seconds=None, max_episode_steps=None, reset=True):
        super(TimeLimitSpe, self).__init__(env)
        self._max_episode_seconds = max_episode_seconds
        self._max_episode_steps = max_episode_steps
        self.need_reset = reset
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
        info["true_angle"] = None

        if done:
            info["over"] = True
        if self._past_limit():  # or done:
            info["past"] = True

        if self.need_reset and (info["past"] or done):
            if isinstance(observation, dict):
                info["true_obs"] = np.copy(observation["observation"])
                info["true_angle"] = observation["angle"]
            else:
                info["true_obs"] = np.copy(observation)
            observation = self.reset()
            done = True
        return observation, reward, done, info

    def reset(self):
        self._episode_started_at = time.time()
        self._elapsed_steps = 0
        obs = self.env.reset()
        return obs


class ContinuousToDiscrete(ActionWrapper):

    def __init__(self, env, args):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(2)
        self.args=args

    def reverse_action(self, action):
        pass

    def action(self, action):
        a = torch.tensor([0] * self.env.action_space.shape[0])
        # if action == 2:
        #     a[7] = 1
        if action == 1:
            a[0] = 0 if not self.args.def_turn else 1
        if action == 0:
            a[0] = -1
        return a


class RestrictContinuous(ActionWrapper):
    def __init__(self, env, args):
        super().__init__(env)
        self.default_actions = build_default_act_space(args)
        size_act = 0
        for a in self.default_actions:
            if not a: size_act += 1
        self.action_space = gym.spaces.Box(np.asarray([-1] * size_act), np.asarray([1] * size_act))

    def reverse_action(self, action):
        pass

    def action(self, action):
        a = torch.tensor([0.] * self.env.action_space.shape[0])
        cursor = 0
        for p in range(self.env.action_space.shape[0]):
            if not self.default_actions[p]:
                a[p:p + 1] = action[cursor]
                cursor += 1
        return a


class OrdRestrictContinuous(ActionWrapper):
    def __init__(self, env, args, force=False, n=10):
        super().__init__(env)
        self.default_actions = build_default_act_space(args)
        self.force = force
        if force and not self.args.def_turn:
            self.default_actions[0]=True
            self.default_actions[7]=True

        size_act = 0
        for a in self.default_actions:
            if not a: size_act += 1
        self.action_space = gym.spaces.Box(np.asarray([-1] * size_act), np.asarray([1] * size_act))
        self.args = args
        self.agent = Nord(False, n, self.env.action_space)

    def reverse_action(self, action):
        pass

    def action(self, action):
        a = torch.tensor([0.] * self.env.action_space.shape[0])
        cursor = 0
        for p in range(self.env.action_space.shape[0]):
            if self.force and not self.args.def_turn and p == 0:
                continue
            if not self.default_actions[p]:
                a[p:p + 1] = action[cursor]
                cursor += 1
        a_ord = self.agent.act(self.args)
        a[0:1] = a_ord[0:1]
        a[7:8] = a_ord[7:8]
        return a


class WrapPyTorch(Wrapper):
    def __init__(self, env, squeeze=False):
        """Return only every `skip`-th frame"""
        super(WrapPyTorch, self).__init__(env)
        self.device = "cpu"
        self.squeeze = squeeze
        self.metadata = {
            'render.modes': ['rgb_array'],
        }

    def reset(self):
        obs = self.env.reset().copy()
        # if obs.dtype != np.uint8:
        if isinstance(obs, dict):
            dtype = torch.uint8 if obs["observation"].dtype != np.float and obs[
                "observation"].dtype != np.float32 else torch.float
            obs["observation"] = torch.from_numpy(obs["observation"]).to(dtype=dtype).view(
                (1,) + obs["observation"].shape)
        else:
            dtype = torch.uint8 if obs.dtype != np.float and obs.dtype != np.float32 else torch.float
            obs = torch.from_numpy(obs).unsqueeze(0).to(dtype=dtype)
        return obs

    def step(self, actions):
        if not self.squeeze:
            actions = actions.view(-1).cpu().numpy()  # remove the squeeze
        else:
            actions = actions.item()
        obs, reward, done, info = self.env.step(actions)
        obs = obs.copy()
        if isinstance(obs, dict):
            dtype = torch.uint8 if obs["observation"].dtype != np.float and obs[
                "observation"].dtype != np.float32 else torch.float
            obs["observation"] = torch.from_numpy(obs["observation"]).to(self.device, dtype=dtype).view(
                (1,) + obs["observation"].shape)
            if not ("true_obs" in info) or info["true_obs"] is None:
                info["true_obs"] = None
            else:
                info["true_obs"] = torch.from_numpy(info["true_obs"]).to(self.device, dtype=dtype).view(
                    obs["observation"].shape).squeeze(0)
        else:
            dtype = torch.uint8 if obs.dtype != np.float and obs.dtype != np.float32 else torch.float
            obs = torch.from_numpy(obs).to(device=self.device, dtype=dtype).unsqueeze(0)
            info["true_obs"] = None if not ("true_obs" in info) or info["true_obs"] is None else torch.from_numpy(
                info["true_obs"]).to(self.device, dtype=dtype)
        # reward  = torch.from_numpy(reward).float().to(self.device)
        reward = torch.tensor(reward).float().unsqueeze(-1).to(self.device)

        return obs, reward, done, info

    def seed(self, seed=None):
        self.env.seed(seed)