import copy
import random

import gym
import torch.distributions
from torch import optim

from models.simple_agents import Pord
from tools.utils import soft_update


class Dqn:
    def __init__(self, args, critic,act_space):
        self.act_space = gym.spaces.Discrete(2)
        self.args=args
        self.critic = critic
        self.optimizer = optim.Adam(self.critic.parameters(), args.lr, eps=1e-6, weight_decay=1e-6)
        self.target_critic = copy.deepcopy(self.critic)
        for param_target in  self.target_critic.parameters():
            param_target.requires_grad = False  # not update by gradient
        self.loss, self.q_value = 0, 0
        self.cpt_fix = 0
        self.cpt_all = 0
        self.true_agent = None

    def act(self, obs, rand=False, **kwargs):
        if rand:
            self.p_a = None
            return torch.tensor(self.act_space.sample())
        q_values = self.critic(obs)
        dist = torch.distributions.Categorical(logits=self.args.alpha*q_values)
        if random.random() < 0.01:
            action = torch.tensor([self.act_space.sample()]).to(self.args.device)
        else:
            action = dist.sample()
        self.p_a = torch.gather(dist.probs, 1, action.unsqueeze(0)).detach().cpu()


        self.cpt_all += 1
        if action == 1:
            self.cpt_fix += 1
        return action.detach()

    def get_value(self, sample, **kwargs):
        # next_obs = preprocess(sample["next_obs"],self.args)
        next_obs = sample["embed"]
        next_q_values = self.target_critic(next_obs)
        if self.args.double:
            on_next_q_values = self.critic(next_obs)
            ind_a = torch.max(on_next_q_values, dim=1, keepdim=True)[1]
            max_next_q_values = torch.gather(next_q_values,1,ind_a)
        else:
            max_next_q_values = torch.max(next_q_values, dim=1, keepdim=True)[0]
        return max_next_q_values

    def evaluate(self,sample,ret):
        obs, action = sample["prev_embed"], sample["actions"]
        action = (sample["actions"][:,0] == (1 if self.args.def_turn else 0)).to(torch.long).view(-1,1)
        # self.q_values = self.critic(obs.detach())
        q_values = self.critic(obs.detach()) if self.args.drl_inputs != "mix" else self.critic(obs)

        # q_value = q_values[action.squeeze().to(self.args.device)].squeeze()
        # q_value = torch.gather(self.q_values, 1, action.to(self.args.device))
        q_value = torch.gather(q_values, 1, action)
        self.optimizer.zero_grad()
        value_loss_all = torch.nn.functional.smooth_l1_loss(q_value.view(-1), ret.detach().view(-1), reduction='none')
        value_loss = value_loss_all.mean()
        if self.args.drl_inputs == "mix":
            value_loss.backward(retain_graph=True)
        else:
            value_loss.backward()
        self.optimizer.step()

        soft_update(self.target_critic, self.critic, self.args.tau)

        self.loss, self.q_value = value_loss.detach().item(), q_value.mean().detach().item()
        sample["p_new_actions"]= torch.gather(torch.distributions.Categorical(logits=self.args.alpha*q_values).probs, 1, action).detach()

    def log(self, logger):
        logger.log_tabular("losses", self.loss)
        logger.log_tabular("Q_values", self.q_value)
        logger.log_tabular("Fix_ratio", self.cpt_fix/(0.001+self.cpt_all))
        self.cpt_fix = 0
        self.cpt_all = 0


    def save(self, save_dir):
        if self.args.save_model != "null":
            path = save_dir+self.args.save_model + "dqn.pt"
            obj = {}
            obj["network_state_dict"] = self.critic.state_dict()
            if self.target_critic is not None:
                obj["target_network_state_dict"] = self.target_critic.state_dict()
            obj['network_optimizer_state_dict'] = self.optimizer.state_dict()
            torch.save(obj, path)

    def load(self):
        pass
