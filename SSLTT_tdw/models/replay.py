import random
import time

import numpy as np
import torch
import torchvision
from torch.utils.data import BatchSampler, SubsetRandomSampler

from tools.utils import dtype_space, dim_space, get_augmentations


class Replay:

    def __init__(self, args, obs_space, act_space):
        self.args=args
        obs_size = obs_space.shape
        self.act_space = act_space
        dtype_obs = torch.uint8 #dtype_space(obs_space)
        device = "cpu"
        self.next_obs = torch.zeros((args.buffer_size, *obs_size), dtype=dtype_obs, device=device,
                                    requires_grad=False)
        self.obs = torch.zeros((args.buffer_size, *obs_size), dtype=dtype_obs, device=device, requires_grad=False)
        self.rewards = torch.zeros((args.buffer_size, 1), device=device, requires_grad=False)
        self.masks = torch.ones((args.buffer_size, 1), device=device, requires_grad=False)
        self.actions = torch.empty((args.buffer_size, dim_space(act_space)), device=device,dtype=dtype_space(act_space), requires_grad=False)
        self.fixations = torch.empty((args.buffer_size, 1), device=device, requires_grad=False,dtype=torch.bool)

        if self.args.log_backgrounds_probs:
            self.backgrounds = torch.empty((args.buffer_size, 1), device=device, requires_grad=False)
            self.objects = torch.empty((args.buffer_size, 1), device=device, requires_grad=False)
            if self.args.category:
                self.categories = torch.empty((args.buffer_size, 1), device=device, requires_grad=False)
            self.views = torch.empty((args.buffer_size, 1), device=device, requires_grad=False)
            self.prev_views = torch.empty((args.buffer_size, 1), device=device, requires_grad=False)
        # self.ids = torch.empty((args.buffer_size, 1), device=device, requires_grad=False)
        if self.args.importance_sampling:
            self.p_actions = torch.empty((args.buffer_size, 1), device=device,dtype=torch.float32, requires_grad=False)

        if self.args.drl_inputs == "rewards" or self.args.drl_inputs == "prev_rewards" or self.args.drl_inputs == "diff_rewards" or self.args.reward_type == 10:
            self.prev_obs = torch.zeros((args.buffer_size, *obs_size), dtype=dtype_obs, device=device, requires_grad=False)
        if self.args.drl_inputs == "prev_rewards"  or self.args.drl_inputs == "diff_rewards":
            self.prev_prev_obs = torch.zeros((args.buffer_size, *obs_size), dtype=dtype_obs, device=device, requires_grad=False)
            self.prev_actions = torch.empty((args.buffer_size, dim_space(act_space)), device=device,dtype=dtype_space(act_space), requires_grad=False)

        if self.args.augmentations == "standard2" or self.args.augmentations == "combine2":
            self.augmentations = get_augmentations(self.args)
        if "crossmodal" in self.args.regularizers != 1:
            self.use_labels = torch.empty((args.buffer_size, 1), dtype=torch.bool, device=device, requires_grad=False)
        self.index = 0
        self.total_size = 0
        self.cpt_per_objects = None
        self.cpt_per_categories = None

    def reset_sampler(self, several):
        self.sampler = BatchSampler(SubsetRandomSampler(range(self.total_size)), several, drop_last=True)
        self.iterator = iter(self.sampler)

    # pil_new = torchvision.transforms.functional.to_pil_image(next_obs[0])
    # pil_new.show()
    # pil_new = torchvision.transforms.functional.to_pil_image(self.obs[self.index, :])
    # pil_new.show()
    # time.sleep(4)

    def insert(self, obs, next_obs, reward, mask, action, info=None, p_a=None, prev_obs=None, prev_prev_obs = None, prev_action=None):
        if self.cpt_per_objects is None:
            self.cpt_per_objects = np.array([0]*info["total_num_objects"])
            self.cpt_per_categories = np.array([0]*info["total_num_categories"])
        if self.total_size == self.args.buffer_size:
            self.cpt_per_objects[int(self.objects[self.index:self.index+1].item())] -= 1
            self.cpt_per_categories[int(self.categories[self.index:self.index + 1].item())] -= 1
        self.cpt_per_objects[info["oid"]] += 1
        self.cpt_per_categories[info["category"]] += 1
        next_obs = info["true_obs"] if info["true_obs"] is not None else next_obs  # handle reset interactions
        angle = info["true_angle"] if info["true_angle"] is not None else info["angle"]
        if self.args.augmentations == "standard2":
            self.obs[self.index:self.index + 1, :] = self.augmentations(next_obs)
        elif self.args.augmentations == "none":
            self.obs[self.index:self.index + 1, :] = next_obs
        else:
            self.obs[self.index:self.index+1, :] = obs

        if self.args.augmentations != "combine2":
            self.next_obs[self.index:self.index+1, :] = next_obs
        else:
            self.next_obs[self.index:self.index + 1, :] = self.augmentations(next_obs)

        self.rewards[self.index:self.index+1, :] = reward
        self.masks[self.index:self.index+1, :] = mask
        self.actions[self.index:self.index+1, :] = action
        self.fixations[self.index:self.index+1, :] = info["fix"]

        if self.args.log_backgrounds_probs:
            self.backgrounds[self.index:self.index+1, :] = info["position"]
            self.objects[self.index:self.index+1, :] = info["oid"]
        if self.args.category:
            self.categories[self.index:self.index + 1, :] = info["category"]
        if "crossmodal" in self.args.regularizers != 1:
            self.use_labels[self.index:self.index + 1, :] = random.uniform(0,1) <= self.args.plabels
        if self.args.importance_sampling:
            self.p_actions[self.index:self.index+1, :] = p_a
        if self.args.drl_inputs == "rewards" or self.args.drl_inputs == "prev_rewards"  or self.args.drl_inputs == "diff_rewards" or self.args.reward_type == 10:
            self.prev_obs[self.index:self.index + 1, :] = prev_obs
        if self.args.drl_inputs == "prev_rewards" or self.args.drl_inputs == "diff_rewards":
            self.prev_prev_obs[self.index:self.index + 1, :] = prev_prev_obs
            self.prev_actions[self.index:self.index + 1, :] = prev_action
        self.views[self.index:self.index+1,:] = angle
        self.prev_views[self.index:self.index+1,:] = info["prev_angle"]
        # self.ids[self.index:self.index+1, :] = info["oid"]

        self.index += 1
        self.total_size = max(self.total_size, self.index)
        self.index = self.index%self.args.buffer_size

    def sample(self):
        try:
            ind = next(self.iterator)
        except:
            self.reset_sampler(self.args.batch_size)
            ind = next(self.iterator)
        sample = {"obs": self.obs[ind], "next_obs": self.next_obs[ind], "rewards": self.rewards[ind],
                  "masks": self.masks[ind], "actions": self.actions[ind].to(self.args.device)}
        if self.args.log_backgrounds_probs:
            sample["backgrounds"] = self.backgrounds[ind].to(self.args.device)
            sample["objects"] = self.objects[ind].to(self.args.device)
            sample["categories"] = self.categories[ind].to(self.args.device)
        if self.args.importance_sampling:
            sample["p_actions"] = self.p_actions[ind]
        if self.args.drl_inputs == "rewards" or  self.args.drl_inputs == "prev_rewards" or self.args.drl_inputs == "diff_rewards" or self.args.reward_type == 10:
            sample["prev_obs"] = self.prev_obs[ind]
        if self.args.drl_inputs == "prev_rewards" or self.args.drl_inputs == "diff_rewards":
            sample["prev_prev_obs"] = self.prev_prev_obs[ind]
            sample["prev_actions"] = self.prev_actions[ind]
        # if self.args.drl_inputs == "views":
        if "crossmodal" in self.args.regularizers:
            sample["use_labels"] = self.use_labels[ind].to(self.args.device)
        sample["prev_views"] = self.prev_views[ind].to(self.args.device)
        sample["views"] = self.views[ind].to(self.args.device)
        sample["fixations"] = self.fixations[ind].to(self.args.device)
        # sample["ids"]= self.ids[ind]
        return sample

    def save(self, save_dir):
        if self.args.save_buffer != "null":
            path = save_dir+self.args.save_buffer + "buffer.pt"
            obj = {}
            obj["obs"] = self.obs
            obj["next_obs"] = self.next_obs
            obj["actions"] = self.actions
            obj["category"] = self.categories
            obj["objects"] = self.objects
            obj["views"] = self.views
            obj["fixations"] = self.fixations
            torch.save(obj, path)

    def load(self,):
        if self.args.load_buffer != "null":
            path = self.args.load_buffer + "buffer.pt"
            checkpoint = torch.load(path, map_location=torch.device(self.args.device))
            self.obs = checkpoint['obs']
            self.next_obs = checkpoint['next_obs']
            self.actions = checkpoint['actions']
            self.categories = checkpoint['category']
            self.objects = checkpoint['objects']
            self.views = checkpoint['views']
            self.fixations = checkpoint['fixations']
class Queue:
    def __init__(self, args):
        self.max_size = abs(args.queue_size)*args.batch_size*2
        self.queue_t = torch.zeros((self.max_size, args.num_latents), dtype=torch.float, device=args.device, requires_grad=False)
        self.queue_size = 0
        self.K = 0
        self.args=args

    def get_queue(self):
        return self.queue_t[:self.queue_size]

    def queue_unqueue(self, batch):
        self.queue_t[self.K:self.K+batch.shape[0], :] = batch.detach()
        # self.queue_size = min(self.max_size,self.K+batch.shape[0])
        self.queue_size = max(self.queue_size,self.K+batch.shape[0])
        self.K = (self.K+batch.shape[0])%self.max_size
        return self.queue_t[:self.queue_size]
