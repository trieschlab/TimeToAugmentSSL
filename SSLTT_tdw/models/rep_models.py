import copy
import math
import os
from time import sleep

import matplotlib.pyplot
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision import transforms

from models.networks import MLPHead, MLP
from models.replay import Queue
from models.resnets import get_network, get_network_decoder
from tools.utils import preprocess, soft_update, dim_space, encode_actions, size_space, get_augmentations, \
    RewardNormalizer, default_space, get_action_to_predict


class Info():
    def __init__(self,args,mlp):
        super().__init__()
        self.args = args
        self.net = mlp


    def load(self):
        if self.args.load_model != "null":
            path = self.args.load_model + "Predictor.pt"
            checkpoint = torch.load(path, map_location=torch.device(self.args.device))
            self.net.load_state_dict(checkpoint['network_state_dict'])
            if self.net_target is not None:
                self.net_target.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['network_optimizer_state_dict'])

    def save(self, save_dir):
        obj = {}
        if self.args.save_model != "null":
            path = save_dir+self.args.save_model + "Predictor.pt"
            obj = {}
            obj["network_state_dict"] = self.net.state_dict()
            if self.net_target is not None:
                obj["target_network_state_dict"] = self.net_target.state_dict()
            obj['network_optimizer_state_dict'] = self.optimizer.state_dict()
            if self.args.method == 'byol':
                obj["predictor_state_dict"] = self.predictor.state_dict()
            torch.save(obj, path)


class LearnRep(Info):
    """
    Standard predictor which take the predictors from the buffer and use it to elarn them and create the feedback for policies
    Modified from https://github.com/mrernst/CLTT
    """

    #
    def __init__(self, action_space, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.loss_mean=0
        self.action_space =action_space
        if self.args.sim_func == "cosine":
            sim_function = lambda x, x_pair: F.cosine_similarity(x.unsqueeze(1), x_pair.unsqueeze(0), dim=2)
            self.sim_function_simple = lambda x, x_pair, dim=1: F.cosine_similarity(x, x_pair, dim=1)
        elif self.args.sim_func == "dot":
            sim_function = lambda x, x_pair: (x.unsqueeze(1)*x_pair.unsqueeze(0)).sum(dim=2)
            self.sim_function_simple = lambda x, x_pair, dim=1: (x*x_pair).sum(dim=1)
        else:
        # if self.args.sim_func == "euclidian":
            sim_function= lambda x, x_pair: -torch.cdist(x, x_pair)
            self.sim_function_simple = lambda x, x_pair, dim=1: -torch.norm(x-x_pair,2, dim=dim)
        self.actions_to_predict, self.act_to_index = get_action_to_predict(self.args)
        learn_action_size = len(self.actions_to_predict) if len(self.actions_to_predict) > 0 else size_space(action_space)
        self.sim_function = sim_function
        self.net_target = None
        if self.args.method == 'byol' or self.args.reward_type in [1,2,3,4] or self.args.temporal_consistency or self.args.queue_size != 0 or self.args.drl_inputs in ["embeds2","mix2"]:
            self.net_target = get_network(self.args)
            soft_update(self.net_target, self.net, tau=1)
            # self.net_target.eval()
            for param_online, param_target in zip(self.net.parameters(), self.net_target.parameters()):
                param_target.requires_grad = False  # not update by gradient
                param_target.data.copy_(param_online.data)

        if self.args.method == 'byol':
            # self.predictor = MLPHead(self.args.num_latents, 256, self.args.num_latents, batchnorm=True).to(self.args.device)
            self.predictor = MLP(num_inputs=self.args.num_latents, num_output=self.args.num_latents, hidden_size=self.args.neurons_rep,activation=nn.LeakyReLU, num_layers=self.args.layers_rep, last_linear=True, batch_norm=True).to(self.args.device)
            self.optimizer = torch.optim.AdamW(list(self.net.parameters()) + list(self.predictor.parameters()), lr=self.args.lr_rep, weight_decay=self.args.weight_decay)
            self.loss = BYOL_TT_Loss(self.args, self.sim_function_simple)
        # elif self.args.predictor:
        #     self.loss = SimCLR_TT_Loss(sim_function, self.sim_function_simple, self.args)
        #     self.predictor = MLP(num_inputs=self.args.num_latents+size_space(action_space), num_output=self.args.num_latents, hidden_size=self.args.neurons_rep,activation=nn.LeakyReLU, num_layers=self.args.layers_rep, last_linear=True, batch_norm=False).to(self.args.device)
        #     self.optimizer = torch.optim.AdamW(list(self.net.parameters()) + list(self.predictor.parameters()), lr=self.args.lr_rep, weight_decay=self.args.weight_decay)
        else:
            if self.args.method == "simclr":
                self.loss = SimCLR_TT_Loss(sim_function, self.sim_function_simple, self.args)
            elif self.args.method == "vicreg":
                self.loss = VicReg(self.args)
            self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.args.lr_rep, weight_decay=self.args.weight_decay)
            # self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=1e-4, weight_decay=self.args.weight_decay)


        if self.args.predictor:
            if self.args.predictor in [17,18]:
                input_pred = self.args.pred_detach
            elif self.args.equi_proj:
                input_pred = self.args.num_latents
            else:
                input_pred = self.net.repr
            self.predictor = MLP(num_inputs=input_pred+learn_action_size, num_output=input_pred, hidden_size=self.args.neurons_rep,activation=nn.LeakyReLU, num_layers=self.args.layers_rep, last_linear=True, batch_norm=self.args.pred_batchnorm, dropout=self.args.pred_dropout, pre_norm=self.args.pre_norm).to(self.args.device)
            self.optimizer_predictor = torch.optim.AdamW(self.predictor.parameters(), lr=self.args.lr_rep, weight_decay=self.args.weight_decay_pred)
            self.predictor.train()
            if self.args.depth_apart:
                self.depth_predictor = MLP(num_inputs=input_pred+1, num_output=input_pred,hidden_size=self.args.neurons_rep, activation=nn.LeakyReLU,num_layers=self.args.layers_rep, last_linear=True,batch_norm=self.args.pred_batchnorm, dropout=self.args.pred_dropout, pre_norm=self.args.pre_norm).to(self.args.device)
                self.optimizer_depth_predictor = torch.optim.AdamW(self.depth_predictor.parameters(),lr=self.args.lr_rep,weight_decay=self.args.weight_decay_pred)
                self.depth_predictor.train()


        if self.args.inverse:
            if self.args.equi_proj:
                input_pred = self.args.num_latents
            else:
                input_pred = self.net.repr
            self.predictor = MLP(num_inputs=2*input_pred, num_output=learn_action_size, hidden_size=self.args.neurons_rep,activation=nn.LeakyReLU, num_layers=self.args.layers_rep, last_linear=True, batch_norm=self.args.pred_batchnorm, dropout=self.args.pred_dropout, pre_norm=self.args.pre_norm).to(self.args.device)
            self.optimizer_predictor = torch.optim.AdamW(self.predictor.parameters(), lr=self.args.lr_rep, weight_decay=self.args.weight_decay_pred)
            self.predictor.train()
            if self.args.depth_apart:
                self.depth_predictor = MLP(num_inputs=2 * input_pred, num_output=1,hidden_size=self.args.neurons_rep, activation=nn.LeakyReLU,num_layers=self.args.layers_rep, last_linear=True,batch_norm=self.args.pred_batchnorm, dropout=self.args.pred_dropout, pre_norm=self.args.pre_norm).to(self.args.device)
                self.optimizer_depth_predictor = torch.optim.AdamW(self.depth_predictor.parameters(),lr=self.args.lr_rep,weight_decay=self.args.weight_decay_pred)
                self.depth_predictor.train()


        if self.args.queue_size != 0:
            self.queue = Queue(self.args)

        if "relic" in self.args.regularizers:
            self.loss_reg = RELIC_TT_Loss(sim_function)
        if "recons" in self.args.regularizers or "pred_recons" in self.args.regularizers:
            # self.decoder = MLP(num_inputs=self.args.num_latents, num_output=self.args.num_latents, hidden_size=self.args.neurons_rep,activation=nn.LeakyReLU, num_layers=self.args.layers_rep, last_linear=True).to(self.args.device)
            self.decoder = get_network_decoder(self.args).to(self.args.device)
            self.opt_ae = torch.optim.Adam(self.decoder.parameters(), lr=self.args.lr_rep, weight_decay=self.args.weight_decay)
        if "supervised" in self.args.regularizers or "linear_eval" in self.args.regularizers:
            self.num_categories = 104 if self.args.num_obj == 4001 else 105
            self.classifier = nn.Linear(self.args.num_latents,self.num_categories).to(self.args.device)
            # MLP(num_inputs=self.args.num_latents, num_output=104 if args.num_obj == 4001 else 105, hidden_size=self.args.neurons_rep,activation=nn.LeakyReLU, num_layers=self.args.layers_rep, last_linear=True).to(self.args.device)
            self.opt_sup = torch.optim.Adam(self.classifier.parameters(), lr=self.args.lr_rep, weight_decay=self.args.weight_decay)
        if "crossmodal" in self.args.regularizers:
            self.num_categories = 104 if self.args.num_obj == 4001 else 105
            self.label_encoder = MLP(num_inputs=self.num_categories, num_output=self.args.num_latents, hidden_size=self.args.neurons_rep,activation=nn.LeakyReLU, num_layers=self.args.layers_rep, last_linear=True).to(self.args.device)
            self.opt_cross = torch.optim.Adam(self.label_encoder.parameters(), lr=self.args.lr_rep, weight_decay=self.args.weight_decay)
        self.net.train()
        self.old_inputs = None
        self.new_inputs = None
        self.loss_discriminator_mean=None
        self.train_transform = None
        self.distance = None
        self.loss_a = None
        self.loss_recons = None
        if self.args.normalizer:
            self.normalizer = RewardNormalizer()
        self.cpt_up = 0

    def augment(self, x_pair):
        s=1
        if self.train_transform is None:
            self.train_transform = get_augmentations(self.args)
        new_image = self.train_transform(x_pair)
        return new_image

    def embed(self,inputs):
        preprocessed_inputs = preprocess(inputs, self.args)
        e = self.net(preprocessed_inputs)
        if "reconstruction" in self.args.regularizers:
            e = self.encoder(e[0])
            return e, e

        return e

    def target_embed(self,inputs):
        if self.net_target is None:
            return self.embed(inputs)
        preprocessed_inputs = preprocess(inputs, self.args)
        return self.net_target(preprocessed_inputs)

    def drl_embed(self,inputs, embed =None, processed = False, reward=None):
        if self.args.drl_inputs == "ones":
            return torch.ones((inputs.shape[0],self.args.num_latents),device=self.args.device)
        elif self.args.drl_inputs == "rewards" or self.args.drl_inputs == "prev_rewards" or self.args.drl_inputs == "diff_rewards":
            if reward is None:
                raise Exception("Drl inputs rewards not implemented")
            if self.args.normalizer:
                reward = self.normalizer.apply(reward)
            return reward
        elif self.args.drl_inputs == "inputs":
            return preprocess(inputs, self.args) if not processed else inputs
        return embed if embed is not None else self.target_embed(inputs)[0 if self.args.drl_inputs not in ['projmix',"projembeds"] else 1]

    def get_loss(self, e1, e2, neg):
        pos_sim = self.sim_function_simple(e1, e2)/self.args.temperature
        neg_sim = self.sim_function(e1, torch.cat((e2, neg), dim=0) if self.args.neg_pos else neg)/self.args.temperature
        log_neg_sim = -torch.logsumexp(neg_sim, dim=1).view(-1, 1)
        return pos_sim + log_neg_sim

    def learn(self, sample):
        if self.args.augmentations == "standard":
            sample["next_obs"] = sample["next_obs"].to(self.args.device)
            sample["obs"] = sample["next_obs"]
            sample["next_obs"] = self.augment(sample["next_obs"])
        elif self.args.augmentations == "combine":
            sample["obs"] = sample["obs"].to(self.args.device)
            sample["true_next_obs"] = sample["next_obs"].to(self.args.device)
            sample["next_obs"] = self.augment(sample["true_next_obs"])

        ###Representation learning
        prev_inputs, inputs = sample["obs"], sample["next_obs"]
        prev_current_inputs = torch.cat((inputs,prev_inputs),dim=0)
        preprocessed_inputs = preprocess(prev_current_inputs, self.args)
        self.new_inputs, p_prev_inputs = preprocessed_inputs.split(inputs.shape[0])

        if self.args.equi_comb and self.args.augmentations == "combine":
            next_embed = self.net(preprocess(sample["true_next_obs"], self.args))[1 if self.args.equi_proj else 0]

        if not self.args.multiforward:
            embeddings, projection = self.net(preprocessed_inputs)
            pos_projection, prev_projection = projection.split(inputs.shape[0])
            self.new_embed, prev_embed = embeddings.split(inputs.shape[0])
        else:
            self.new_embed, pos_projection = self.net(self.new_inputs)
            prev_embed, prev_projection = self.net(p_prev_inputs)
            projection = torch.cat((pos_projection, prev_projection), dim=0)
            embeddings = torch.cat((self.new_embed, prev_embed), dim=0)
        log_prob = None
        if self.args.projection:
            prev_embed_a = prev_projection
        else:
            prev_embed_a = prev_embed


        if log_prob is None:
            all_neg, prev_target_embed, self.new_target_embed, prev_target_obj = None, None, None, None
            if self.args.queue_size != 0:
                with torch.no_grad():
                    target_embed, proj_target_embed = self.net_target(preprocessed_inputs)
                    self.new_target_embed, prev_target_embed = proj_target_embed.split(inputs.shape[0])
                    if self.args.queue_size > 0:self.queue.queue_unqueue(self.new_target_embed)
                    all_neg = self.queue.get_queue()
                    if self.args.queue_size < 0:self.queue.queue_unqueue(self.new_target_embed)


            if self.args.method == 'byol':
                with torch.no_grad():
                    prev_target_embed, prev_target_proj = self.net_target(preprocessed_inputs) if prev_target_embed is None else (prev_target_embed, prev_target_proj)
                    p1, p2 = prev_target_proj.split(prev_target_proj.shape[0]//2)
                    prev_target_proj = torch.cat((p2,p1),dim=0)
                log_prob, cond_log_prob = self.loss(self.predictor(projection),prev_target_proj, sample, x_neg=self.new_target_embed, x_neg_pair=prev_target_embed, all_neg=all_neg)
            else:
                log_prob, cond_log_prob = self.loss(pos_projection, prev_embed_a, sample, x_neg=self.new_target_embed, x_neg_pair=prev_target_embed, all_neg=all_neg)

        loss = self.args.coef_base*(-log_prob - cond_log_prob)

        self.loss_mean=loss.mean()

        ### Reward computation
        rad = sample["views"] * math.pi / 180
        feedback = 2*(torch.max(torch.abs(torch.cos(rad)), torch.abs(torch.sin(rad)))-0.5)
        self.loss.feedback = feedback.detach()
        self.feedback = feedback.mean().cpu().item()
        feedback = feedback.view(-1, 1)
        return feedback, loss

    def update(self, loss, sample):
        self.cpt_up +=1
        if self.cpt_up%self.args.latency==0:
            if self.args.queue_size and self.queue.queue_size != self.queue.max_size:#<= 2*self.args.batch_size:
                return
            if self.args.importance_sampling:
                ratio = sample["p_new_actions"] / sample["p_actions"].to(self.args.device)
                clip_ratio = torch.clamp(ratio, 1-self.args.clip, 1+self.args.clip)
                loss = clip_ratio*(loss.split(self.args.batch_size)[0])

            loss.mean().backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if "supervised" in self.args.regularizers or "linear_eval" in self.args.regularizers :
                self.opt_sup.step()
                self.opt_sup.zero_grad()

            if "crossmodal" in self.args.regularizers:
                self.opt_cross.step()
                self.opt_cross.zero_grad()

            if "recons" in self.args.regularizers or "pred_recons" in self.args.regularizers:
                self.opt_ae.step()
                self.opt_ae.zero_grad()

            if self.args.predictor or self.args.inverse:
                self.optimizer_predictor.step()
                self.optimizer_predictor.zero_grad()

            if self.args.depth_apart:
                self.optimizer_depth_predictor.step()
                self.optimizer_depth_predictor.zero_grad()


            if self.net_target is not None:
                soft_update(self.net_target, self.net, tau=self.args.tau_target_rep)


    def save(self, save_dir):
        super().save(save_dir)
        if self.args.save_model != "null" and self.args.predictor:
            path = save_dir + self.args.save_model + "APredictor.pt"
            obj = {}
            obj["network_state_dict"] = self.predictor.state_dict()
            obj['network_optimizer_state_dict'] = self.optimizer_predictor.state_dict()
            torch.save(obj, path)
        if self.args.save_model != "null" and "crossmodal" in self.args.regularizers:
            path = save_dir + self.args.save_model + "CrossPredictor.pt"
            obj = {}
            obj["network_state_dict"] = self.label_encoder.state_dict()
            obj['network_optimizer_state_dict'] = self.opt_cross.state_dict()
            torch.save(obj, path)
        if self.args.save_model != "null" and ("recons" in self.args.regularizers or 'pred_recons' in self.args.regularizers):
            path = save_dir + self.args.save_model + "decoder.pt"
            obj = {}
            obj["network_state_dict"] = self.label_encoder.state_dict()
            obj['network_optimizer_state_dict'] = self.opt_cross.state_dict()
            torch.save(obj, path)

    def log(self, logger):
        self.net.eval()
        if self.old_inputs is not None:
            with torch.no_grad():
                new_embed,_ = self.net(self.old_inputs)
                logger.log_tabular("tmp_change",-self.sim_function_simple(self.old_embeds, new_embed).mean().item())
                if self.net_target is not None:
                    new_target_embed, _ = self.net_target(self.old_inputs)
                    logger.log_tabular("target_tmp_change",-self.sim_function_simple(self.old_target_embed, new_target_embed).mean().item())
        else:
            logger.log_tabular("tmp_change",0)
            if self.net_target is not None:
                logger.log_tabular("target_tmp_change",0)
        if self.loss_a is not None:
            logger.log_tabular("loss_a", self.loss_a.item())

        if self.new_inputs is not None:
            self.old_inputs = self.new_inputs.clone().detach()
            self.old_embeds, _ = self.net(self.old_inputs)
            # self.old_embeds = self.new_embed.clone().detach()
            if self.net_target is not None:
                self.old_target_embed ,_ = self.net_target(self.old_inputs)

            if self.loss_discriminator_mean is not None:
                logger.log_tabular("loss_disc", self.loss_discriminator_mean.detach().item())
                logger.log_tabular("loss_ind", self.loss_ind.detach().item())

            if self.loss_recons is not None:
                logger.log_tabular("loss_recons", self.loss_recons.mean().detach().cpu().item())


            logger.log_tabular("rep_loss", self.loss_mean.detach().item())
            logger.log_tabular("distances",  self.loss.positive_distance.item())
            logger.log_tabular("time_distances",  self.distance.mean().item() if self.distance is not None else self.loss.positive_distance.item() )

            logger.log_tabular("norm",  torch.norm(self.old_embeds,dim=1).mean().item())
            logger.log_tabular("irewards",  self.feedback)
        else:
            logger.log_tabular("rep_loss", 0)
            logger.log_tabular("distances", 0)
            logger.log_tabular("norm", 0)
            logger.log_tabular("irewards", 0)
        if self.args.normalizer:
            logger.log_tabular("var",  self.normalizer.var)
            logger.log_tabular("mean",  self.normalizer.mean)

        self.net.train()


class RELIC_TT_Loss(nn.Module):
    """
        RELIC loss which minimizes similarities the same between the anchor and different views of other samples,
            i.e. x and its pair x_pair
    """

    def __init__(self, sim_func):
        """Initialize the RELIC_TT_Loss class"""
        super(RELIC_TT_Loss, self).__init__()
        self.sim_func = sim_func


    def forward(self, x, x_pair):
        """
        params:
            x: representation tensor (Tensor)
            x_pair: tensor of the same size as x which should be the pair of x (Tensor)
        return:
            loss: the loss of RELIC-TT (Tensor)
        """
        sim = self.sim_func(x, x_pair)
        loss = F.kl_div(sim.softmax(-1).log(), sim.T.softmax(-1), reduction='none')
        return loss


class VicReg(nn.Module):
    """
        BYOL loss that maximizes cosine similarity between the online projection (x) and the target projection(x_pair)
    """

    def __init__(self, args):
        """Initialize the SimCLR_TT_Loss class"""
        super(VicReg, self).__init__()
        self.args=args
        self.positive_samples, self.negatives_only, self.no_exp_log_prob, self.lower_tmp, self.higher_tmp, self.feedback = None, None, None, None, None, None


    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x, y, *args, **kwargs):

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        repr_loss = F.mse_loss(x, y)

        # repr_loss = F.mse_loss(x, y, reduction="none")

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        # std_loss = F.relu(1 - std_x) / 2 + F.relu(1 - std_y) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(self.args.num_latents) + self.off_diagonal(cov_y).pow_(2).sum().div(self.args.num_latents)
        # cov_loss = self.off_diagonal(cov_x).pow_(2).div(self.args.num_latents) + self.off_diagonal(cov_y).pow_(2).div(self.args.num_latents)

        self.positive_distance = repr_loss.detach().mean()
        self.log_prob = (- cov_loss - std_loss).detach()
        self.log_cond_prob = -repr_loss.detach()

        # return - cov_loss - 25* std_loss, -25*repr_loss
        return - cov_loss - std_loss, - repr_loss

class BYOL_TT_Loss(nn.Module):
    """
        BYOL loss that maximizes cosine similarity between the online projection (x) and the target projection(x_pair)
    """

    def __init__(self, args,sim_func_simple):
        """Initialize the SimCLR_TT_Loss class"""
        super(BYOL_TT_Loss, self).__init__()
        self.args=args
        self.positive_samples, self.negatives_only, self.no_exp_log_prob, self.lower_tmp, self.higher_tmp, self.feedback = None, None, None, None, None, None
        self.sim_func_simple = sim_func_simple

    def forward(self, x, x_target, *args, **kwargs):
        """
        params:
            x: representation tensor (Tensor)
            x_pair: tensor of the same size as x which should be the pair of x (Tensor)
        return:
            loss: the loss of BYOL-TT (Tensor)
        """
        # x = F.normalize(x, dim=-1, p=2)
        # x_target = F.normalize(x_target, dim=-1, p=2)
        # loss = 2-2*self.sim_func(x, x_target).diag()
        # x_mix = torch.cat((x[:x.shape[0]//2],x_target[x.shape[0]//2:]),dim=0)
        # y = torch.cat((x_target[:x.shape[0]//2],x[x.shape[0]//2:]),dim=0)
        # loss = 2-2*self.sim_func_simple(x_mix, y)

        loss = 2-2*self.sim_func_simple(x, x_target)#/self.args.temperature
        self.log_prob = -loss.detach()[:self.args.batch_size]
        self.log_cond_prob = -loss.detach()[:self.args.batch_size]
        self.positive_distance = loss.detach().mean()

        return -loss, 0

class SimCLR_TT_Loss(nn.Module):
    def __init__(self, sim_func, simple_sim_func, args):
        """Initialize the SimCLR_TT_Loss class"""
        super(SimCLR_TT_Loss, self).__init__()
        self.args = args
        self.simple_sim_func = simple_sim_func
        self.batch_size=args.batch_size
        self.mask = self.mask_correlated_samples(self.args.batch_size)
        self.double_mask = torch.ones(2 * args.batch_size, 2 * args.batch_size, dtype=torch.bool)
        self.double_mask = self.double_mask.fill_diagonal_(0)
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.sim_func = sim_func
        self.log_cond_prob = 0
        self.log_prob = 0
        self.loss_a = None
        self.feedback = None

        self.positive_samples, self.negatives_only, self.no_exp_log_prob, self.lower_tmp, self.higher_tmp, self.feedback = None, None, None, None, None, None

    def mask_correlated_samples(self, batch_size):
        """
        mask_correlated_samples takes the int batch_size
        and returns an np.array of size [2*batchsize, 2*batchsize]
        which masks the entries that are the same image or
        the corresponding positive contrast
        """
        if self.args.no_double:
            mask = torch.ones(batch_size, batch_size, dtype=torch.bool)
            mask = mask.fill_diagonal_(0)
            return mask

        mask = torch.ones(2 * batch_size, 2 * batch_size, dtype=torch.bool)
        mask = mask.fill_diagonal_(0)

        # fill off-diagonals corresponding to positive samples
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def compute_mask(self, labels, sim):
        N = 2 * self.batch_size
        mask = torch.ones(2 * self.batch_size, 2 * self.batch_size, dtype=torch.bool)
        mask = mask.fill_diagonal_(0)

        labels = torch.cat((labels, labels), dim=0).squeeze()
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # fill off-diagonals corresponding to positive samples
        loss = 0
        for i in range(self.batch_size*2):
            m = (labels[i] != labels)
            loss += -positive_samples[i] + torch.log(torch.exp(positive_samples[i]).detach() + torch.exp(sim[i][m]).sum())
            # mask[i:i+1][m] = 0
            # mask[i][m] = 0
            # mask[self.batch_size + i][m] = 0
        self.positive_distance = positive_samples.detach().mean()

        return loss / (2*self.batch_size)


    def get_simclr(self,x, x_pair):
        z = torch.cat((x, x_pair), dim=0)
        sim = self.sim_func(z, z) / self.args.temperature
        # if labels is not None:
        #     return self.compute_mask(labels, sim)

        N = 2 * self.batch_size
        # get the entries corresponding to the positive pairs
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)


        # we take all of the negative samples
        negative_samples = sim[self.mask].reshape(N, -1)
        if self.args.number_negatives:
            # if we specify N negative samples: do random permutation of negative sample losses,
            # such that we do consider different positions in the batch
            negative_samples = torch.take_along_dim(
                negative_samples, torch.rand(*negative_samples.shape, device=self.args.device).argsort(dim=1), dim=1)
            # cut off array to only consider N_negative samples per positive pair
            negative_samples = negative_samples[:, :self.args.number_negatives]
            # so what we are doing here is basically using the batch to sample N negative
            # samples.
        if self.args.neg_pos == 1:
            logits = torch.cat((positive_samples, negative_samples), dim=1)
        elif self.args.neg_pos == 2:
            logits = torch.cat((self.simple_sim_func(x,x)/self.args.temperature, negative_samples), dim=1)
        else:
            logits = negative_samples
        return positive_samples, negative_samples, logits

    def get_simclr_v2(self,x, x_pair):
        #Changer x_pair qui est le précéent en x
        sim = self.sim_func(x, x_pair) if not self.args.predictor else self.sim_func(x_pair, x)
        sim = sim / self.args.temperature

        N = self.batch_size
        # get the entries corresponding to the positive pairs
        # sim_i_j = torch.diag(sim, self.batch_size)
        sim_i_j = torch.diag(sim)
        positive_samples = sim_i_j.reshape(N, 1)

        # we take all of the negative samples
        negative_samples = sim[self.mask[:sim.shape[0]]].reshape(N, -1)
        if self.args.number_negatives:
            # if we specify N negative samples: do random permutation of negative sample losses,
            # such that we do consider different positions in the batch
            negative_samples = torch.take_along_dim(
                negative_samples, torch.rand(*negative_samples.shape, device=self.args.device).argsort(dim=1), dim=1)
            # cut off array to only consider N_negative samples per positive pair
            negative_samples = negative_samples[:, :self.args.number_negatives]
            # so what we are doing here is basically using the batch to sample N negative
            # samples.
        if self.args.neg_pos == 1:
            logits = torch.cat((positive_samples, negative_samples), dim=1)
        elif self.args.neg_pos == 2:
            logits = torch.cat((self.simple_sim_func(x,x).unsqueeze(1)/self.args.temperature, negative_samples), dim=1)
        else:
            logits = negative_samples


        return positive_samples, negative_samples, logits


    def get_moco(self,x, x_pair, x_neg, x_neg_pair, all_neg):
        if not self.args.no_double:
            z = torch.cat((x, x_pair), dim=0)
            z_pos = torch.cat((x_neg_pair, x_neg), dim=0)
        elif not self.args.predictor:
            z = x
            z_pos = x_neg_pair
        else:
            z = x_pair
            z_pos = x_neg

        # all_neg = self.queue.get_queue()
        positive_samples = self.simple_sim_func(z, z_pos)/ self.args.temperature
        negative_samples = self.sim_func(z, all_neg)/ self.args.temperature
        # self.queue.queue_unqueue(z_pos)

        return positive_samples, negative_samples, negative_samples

    def forward(self, x, x_pair, sample=None, x_neg=None, x_neg_pair=None, all_neg=None,  labels=None, store=True):
        """
        Given a positive pair, we treat the other 2(N − 1)
        augmented examples within a minibatch as negative examples.
        to control for negative samples we just cut off losses
        """
        if x_neg is None:
            if self.args.no_double:
                positive_samples, negative_samples, logits = self.get_simclr_v2(x, x_pair)
            else:
                positive_samples, negative_samples, logits = self.get_simclr(x, x_pair)
        else:
            positive_samples, negative_samples, logits = self.get_moco(x, x_pair, x_neg, x_neg_pair, all_neg)



        if store: self.positive_distance = positive_samples.detach().mean()

        # the following is more or less a trick to reuse the cross-entropy function for the loss
        # Think of the loss as a multi-class problem and the label is 0
        # such that only the positive similarities are picked for the numerator
        # and everything else is picked for the denominator in cross-entropy
        # labels = torch.zeros(N).to(self.args.device).long()
        # loss = self.criterion(logits, labels)
        log_cond_prob = positive_samples.squeeze() #Maximize negative distance; high when close to 0
        # log_prob = -torch.log(torch.sum(torch.exp(logits), dim=1)).squeeze() #Maximize distance between a sample and its negatives, high when far from others
        log_prob = -torch.logsumexp(logits, dim=1).squeeze() #Maximize distance between a sample and its negatives, high when far from others
        if self.args.log_backgrounds_probs and sample is not None and store:
            self.positive_samples=positive_samples.detach()
            self.log_cond_prob = positive_samples.split(self.batch_size)[0].detach()
            self.log_prob = log_prob.split(self.batch_size)[0].detach()
            self.negatives_only = -torch.logsumexp(negative_samples, dim=1).squeeze().split(self.batch_size)[0].detach()
            self.no_exp_log_prob = -logits.sum(dim=1).squeeze().detach()
            self.lower_tmp = -torch.logsumexp(logits/0.1, dim=1).squeeze().detach()
            self.higher_tmp = -torch.logsumexp(logits/10, dim=1).squeeze().detach()
        # self.backgrounds = sample["backgrounds"].squeeze().detach()
        # self.objects = sample["objects"].squeeze().detach()

        return log_prob, log_cond_prob

