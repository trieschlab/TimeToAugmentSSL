import math
import random
import time

import numpy as np
import torch

from tools.utils import build_default_act_space


class BasicAgent:

    def __init__(self, action_space):
        self.action_space = action_space
        self.p_a = 1


    def get_value(self, sample, **kwargs):
        return 0

    def log(self, logger):
        pass

    def evaluate(self, sample, ret):
        pass

    def save(self, save_dir):
        pass

    def load(self):
        pass


class Nfix(BasicAgent):
    def __init__(self, nfix, action_space):
        super().__init__(action_space)
        self.nfix = nfix
        self.cpt = 0

    def act(self, *args, **kwargs):
        self.cpt += 1
        if self.cpt%self.nfix != 0:
            return torch.tensor([2]).cpu()
        return torch.tensor([self.action_space.sample()])


class FreqAgentCat(BasicAgent):
    def __init__(self, buffer, action_space):
        super().__init__(action_space)
        self.buffer=buffer
        assert self.action_space.__class__.__name__ == 'Box', "wrong action space"

    def act(self, *args, full_obs=None, **kwargs):
        if self.buffer.cpt_per_objects is not None:
            freqs = self.buffer.cpt_per_categories/self.buffer.total_size
            deciles = np.quantile(freqs, np.arange(0.1, 1.1, 0.1))
            # print( (freqs[full_obs["oid"]] <= deciles))
            # print( (freqs[full_obs["oid"]] <= deciles).nonzero())
            # print( (freqs[full_obs["oid"]] <= deciles).nonzero()[0])

            num = 2. + 9. - (freqs[full_obs["category"]] <= deciles).nonzero()[0][0].item()
        else:
            num = 2 + 9

        a = torch.tensor(self.action_space.sample()).cpu()
        if random.uniform(0,1) < 1/num:
            a[0:1] = -1
        else:
            a[0:1] = 0
        return a

class FreqAgentObj(BasicAgent):
    def __init__(self, buffer, action_space):
        super().__init__(action_space)
        self.buffer=buffer
        assert self.action_space.__class__.__name__ == 'Box', "wrong action space"

    def act(self, *args, full_obs=None, **kwargs):
        if self.buffer.cpt_per_objects is not None:
            freqs = self.buffer.cpt_per_objects/self.buffer.total_size
            deciles = np.quantile(freqs, np.arange(0.1, 1.1, 0.1))
            # print( (freqs[full_obs["oid"]] <= deciles))
            # print( (freqs[full_obs["oid"]] <= deciles).nonzero())
            # print( (freqs[full_obs["oid"]] <= deciles).nonzero()[0])

            num = 2. + 9. - (freqs[full_obs["oid"]] <= deciles).nonzero()[0][0].item()
        else:
            num = 2 + 9

        a = torch.tensor(self.action_space.sample()).cpu()
        if random.uniform(0,1) < 1/num:
            a[0:1] = -1
        else:
            a[0:1] = 0
        return a

class NordPturn(BasicAgent):
    def __init__(self, nturn, pfix, action_space):
        super().__init__(action_space)
        self.pfix = pfix
        self.nturn = nturn
        self.cpt = 0
        assert self.action_space.__class__.__name__ == 'Box', "wrong action space"

    def act(self, *args, **kwargs):
        self.cpt += 1
        if self.cpt%self.nturn != 0:
            act0 = 0 #if not self.args.def_turn else 1
        else:
            act0 = -1

        if random.random() < self.pfix:
            act_fix= -1
        else:
            act_fix = 1
        a = torch.tensor(self.action_space.sample()).cpu()
        a[0:1] = act0
        a[-1] = act_fix
        return a

class Pord(BasicAgent):
    def __init__(self, def_turn, nfix, action_space):
        super().__init__(action_space)
        self.nfix = float(nfix)
        self.def_turn=def_turn

    def _cont_act(self):
        a = torch.tensor(self.action_space.sample()).cpu()
        if random.uniform(0,1) < 1./self.nfix:
            # a[1:2] = -1
            a[0:1] = 0 if not self.def_turn else -1
            a[7:8] = 1

            return a
        # a[1:2] = 1
        a[0:1] = -1
        a[7:8] = -1
        if self.def_turn: a[0:1] = - a[0:1]
        return a

    def act(self, *args, **kwargs):
        if self.action_space.__class__.__name__ == 'Box':
            return self._cont_act()
        if random.uniform(0,1) < 1./self.nfix:
            return torch.tensor([2]).cpu()
        return torch.tensor([0]).cpu()

class Nord(BasicAgent):
    def __init__(self, def_turn, nfix, action_space):
        super().__init__(action_space)
        self.nfix = nfix
        self.cpt = 0
        self.def_turn=def_turn

    def _cont_act(self):
        self.cpt += 1
        a = torch.tensor(self.action_space.sample()).cpu()
        if self.def_turn:
            return a
        if self.cpt%self.nfix != 0:
            a[0:1] = 0
            a[7:8] = 1
            return a
        a[0:1] = -1
        a[7:8] = -1
        return a

    def act(self, *args, **kwargs):
        if self.action_space.__class__.__name__ == 'Box':
            return self._cont_act()

        self.cpt += 1
        if self.cpt%self.nfix != 0:
            return torch.tensor([2]).cpu()
        return torch.tensor([0]).cpu()

class CurrOrd(BasicAgent):
    def __init__(self, action_space):
        super().__init__(action_space)
        self.agent = Nord(1, action_space)

    def act(self, *args, steps=0, **kwargs):
        if steps != 0 and steps%10000 == 0 and steps <= 100000:
            self.agent = Nord(steps // 10000, self.action_space)
        return self.agent.act(*args,**kwargs)

class Left(BasicAgent):
    def __init__(self, action_space):
        super().__init__(action_space)

    def act(self, *args, **kwargs):
        return torch.tensor([0]).cpu()


class Rnd(BasicAgent):
    def __init__(self, action_space):
        super().__init__(action_space)

    def act(self, *args, **kwargs):
        return torch.tensor([self.action_space.sample()])

class Play(BasicAgent):
    def __init__(self, action_space):
        super().__init__(action_space)
        self.mapping = {"left": 0, "right": 1, "up": 2}

    def act(self, *args, **kwargs):
        import keyboard
        time.sleep(0.2)
        k=keyboard.read_key()
        return torch.tensor([self.mapping[k]])

class HandDefined:
    def __init__(self, args):
        self.args = args
        self.default=build_default_act_space(args)
        self.val_pitch = None
        self.val_roll = None
        self.size_act = 0
        for a in self.default:
            if not a: self.size_act += 1
        # self.break_after_one = (self.args.aug_meth == "break")

    def act(self, a):
        action = a
        #Discrete agent
        if self.args.agent == "dqn":
            action = torch.tensor([0.] * self.size_act)
            if a == 1:
                action[0:1] = 0 if not self.args.def_turn else 1
            if a == 0:
                action[0:1] = -1
            # while i < len(self.default):
            #     if i != 0 and not self.default[i]:
            #         action[index] = random.uniform(-1,1)
            #         index+=1
            #     i=i+1

        i=0
        index = 0
        #Continuous agent
        while i < len(self.default):
            # if not self.default[i]:
            #     action[index] = random.uniform(-1,1)

            if i == 2 and self.args.noise < 0:
                if random.random() < abs(self.args.noise):
                    angle = random.uniform(0, 2 * math.pi)
                    action[index], action[index+1] = math.cos(angle), math.sin(angle)
                else:
                    action[index], action[index+1] = 0, 0
                index += 1
                i += 1

            elif i == 6 and self.args.focus < 0:
                if random.random() < abs(self.args.focus):
                    action[index] = random.uniform(-1, 1)
                else:
                    action[index] = -2 if self.args.focus_depth else -0.5
            elif i == 8 and self.args.pitch < 0:
                if random.random() < abs(self.args.pitch):
                    self.val_pitch = random.uniform(-1, 1)
                    action[index] = self.val_pitch
                else:
                    action[index] = -1 if self.args.min_angle_speed == 0 else 0
            elif i == 9 and self.args.roll < 0:
                if random.random() < abs(self.args.roll):
                    self.val_roll = random.uniform(-1, 1)
                    action[index:index+1] = self.val_roll
                else:
                    action[index:index+1] = -1 if self.args.min_angle_speed == 0 else 0
            if not self.default[i]:
                index = index +1
            i+=1
        return action