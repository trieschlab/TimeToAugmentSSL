import copy
import random

import torch

from models.simple_agents import Nord
from tools.utils import soft_update, dim_space


class SAC():
    """
    Standard soft actor critic with goal concatenation
    """
    def __init__(self,args,action_space,critic,actor,critic2=None, fix=0):
        self.args=args
        self.action_space=action_space
        self.act_dim = dim_space(action_space)
        self.critic=critic
        self.critic2=critic2
        if args.double_critic:
            self.target_critic2=copy.deepcopy(critic2)
            self.optimizer_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=args.lr, weight_decay=1e-6)

        self.target_critic = copy.deepcopy(critic)
        self.optimizer_critic=torch.optim.Adam(self.critic.parameters(), lr=args.lr, weight_decay=1e-6)
        self.actor=actor
        self.optimizer_actor=torch.optim.Adam(self.actor.parameters(), lr=args.lr, weight_decay=1e-6)



    def act(self, obs, rand=False, **kwargs):
        if rand:
            self.p_a = None
            return torch.tensor(self.action_space.sample(), device = self.args.device)

        action, log_pb_a, _ = self.actor(obs)
        action = action.view(-1)
        self.p_a = torch.exp(log_pb_a)
        # if self.fix_agent is not None:
        #     action2 = self.fix_agent.act(obs, rand, **kwargs)
        #     action = torch.cat((action2[0:1], action),dim=0)
            # action[-1:] = action2[-1:]
        return action.detach().view(-1)


    def get_value(self, sample, **kwargs):
        next_obs = sample["embed"]

        action, action_log_probs, _ = self.actor(next_obs, deterministic=False)
        # inputs_critic = torch.cat((next_obs, action), dim=1)
        values = self.target_critic.forward_double(next_obs, action)
        if self.args.double_critic:
            values2 = self.target_critic2.forward_double(next_obs, action)
            values = torch.min(values, values2)
        values_ent= values - action_log_probs/self.args.alpha
        return values_ent.view(-1)

    def evaluate(self,sample,ret):
        obs, action = sample["prev_embed"], sample["actions"]
        obs = obs if self.args.drl_inputs == "mix" else obs.detach()

        # inputs_critic = torch.cat((obs, action), dim=1)
        # pi_input_critic=torch.cat((obs.detach(), best_action), dim=1)

        ###Critic optimization
        cval_loss,values=self.optimize_critic(obs, action, ret)
        # cval_loss, values = 0,0
        self.loss, self.q_value = cval_loss.detach().item(), values.detach().item()

        ###Actor optimization
        best_action, best_action_log_prob, _ = self.actor(obs)
        val=self.critic.forward_double(obs.detach(), best_action).view(-1)

        if self.args.double_critic:
            val2=self.critic2.forward_double(obs.detach(), best_action).view(-1)
            val = torch.min(val,val2)

        self.optimizer_actor.zero_grad()
        pi_loss=(-val + best_action_log_prob.view(-1) / self.args.alpha).mean()
        pi_loss.backward() #+ self.entropy_bonus*dist.entropy().sum()
        # if self.args.clip_grad_sac:
        #     torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.clip_grad_sac)
        self.optimizer_actor.step()

        return cval_loss,values,pi_loss


    def optimize_critic(self,obs, action,target):
        values = self.critic.forward_double(obs, action).view(-1)
        self.optimizer_critic.zero_grad()
        cval_loss = torch.nn.functional.smooth_l1_loss(values, target, reduction="mean")
        cval_loss.backward()
        # if self.args.clip_grad_sac:
        #     torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.clip_grad_sac)
        self.optimizer_critic.step()
        soft_update(self.target_critic, self.critic, self.args.tau)

        if self.args.double_critic:
            values2 = self.critic2.forward_double(obs, action).view(-1)
            self.optimizer_critic2.zero_grad()
            torch.nn.functional.smooth_l1_loss(values2, target, reduction="mean").backward()
            if self.args.clip_grad_sac:
                torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.args.clip_grad_sac)
            self.optimizer_critic2.step()
            soft_update(self.target_critic2, self.critic2, self.args.tau)
        return cval_loss.detach(),values.mean().detach()

    def load(self):
        pass
        # if self.context.load_model_path:
        #     path = self.context.load_model_path  + "SAC.pt"
        #     checkpoint=torch.load(path,map_location=torch.device(self.args.device) )
        #     self.actor.load_state_dict(checkpoint['actor_state_dict'])
        #     self.optimizer_critic.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        #     self.optimizer_actor.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        #     for param_group in self.optimizer_actor.param_groups:
        #         param_group["lr"]=self.args.lr_sac
        #     for param_group in self.optimizer_critic.param_groups:
        #         param_group["lr"]=self.args.lr_sac
        #     self.critic.load_state_dict(checkpoint['critic_state_dict'])
        #     self.target_critic.load_state_dict(checkpoint['target'])
        #     if self.args.double_critic:
        #         self.optimizer_critic2.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        #         self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        #         self.target_critic2.load_state_dict(checkpoint['target2'])
        #         for param_group in self.optimizer_critic2.param_groups:
        #             param_group["lr"] = self.args.lr_sac

    def save(self, save_dir):
        if self.args.save_model != "null":
            path = save_dir+self.args.save_model + "dqn.pt"
            obj = {}
            obj["network_state_dict"] = self.critic.state_dict()
            obj["actor_state_dict"] = self.actor.state_dict()
            if self.target_critic is not None:
                obj["target_network_state_dict"] = self.target_critic.state_dict()
            obj['network_optimizer_state_dict'] = self.optimizer_critic.state_dict()
            obj['actor_optimizer_state_dict'] = self.optimizer_actor.state_dict()
            torch.save(obj, path)

    def log(self, logger):
        logger.log_tabular("losses", self.loss)
        logger.log_tabular("Q_values", self.q_value)
        self.cpt_fix = 0
        self.cpt_all = 0

