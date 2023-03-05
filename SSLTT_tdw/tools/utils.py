import math
import sys

import logging
import os
import random

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
import torch.nn.functional as F

from envs.objects import Objects20, Objects5, Objects40, Objects80, Objects1, Objects4k_untex, Objects4k_tex, \
    Objects4k_untex_small


def size_space(action_space):
    return action_space.n if action_space.__class__.__name__ == "Discrete" else action_space.shape[0]

def dim_space(action_space):
    return 1 if action_space.__class__.__name__ == 'Discrete' else action_space.shape[0]

def dtype_space(action_space):
    return torch.long if action_space.__class__.__name__ == 'Discrete' else torch.float32

def default_space(args, action_space, n):
    if args.min_angle_speed == args.max_angle_speed:
        return torch.zeros((n, action_space.shape[0]))
    elif args.min_angle_speed == 0:
        return torch.zeros((n, action_space.shape[0]))-1
    else:
        raise Exception("Not possible with these angle speeds")

def get_action_to_predict(args):
    default = build_default_act_space(args)
    index = 0
    indexes = []
    act_to_index = {}
    for i in range(len(default)):
        if default[i]:
            continue
        act_to_index[i] = index
        if i == 4 and args.depth_apart:
            continue
        if str(i) in args.action_predict:
            indexes.append(index)
        index += 1
    return torch.tensor(indexes, device=args.device,dtype=torch.long), act_to_index


def encode_actions(action, action_space):
    return F.one_hot(action.squeeze(), action_space.n) if action_space.__class__.__name__ == "Discrete" else action

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

class RewardNormalizer:
    def __init__(self):
        self.var = 1.
        self.mean = 0.
        self.alpha = 0.001

    def update(self, x):
        x = x.detach()
        with torch.no_grad():
            mu = x.mean()
            var = x.var()
            if self.var == 1. and self.mean == 0.:
                self.mean = mu
                self.var = var
            else:
                self.var = (1-self.alpha)*self.var + self.alpha*var + (1- self.alpha)*self.alpha*(mu - self.mean)**2
                self.mean = (1-self.alpha)*self.mean + self.alpha*mu
        norm_x = (x - self.mean)/math.sqrt(self.var)
        return norm_x

    def apply(self,x):
        return (x.detach() - self.mean)/math.sqrt(self.var)

def setup_logger(name,save_dir=None, log_file="/logs.log", level=logging.INFO,out=True,formatter=None):
    """Function setup as many loggers as you want"""
    logger = logging.getLogger(name)
    logger.handlers=[]
    formatter = logging.Formatter('%(asctime)s -- %(levelname)s -- %(message)s') if formatter is None else logging.Formatter(formatter)
    if out:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if save_dir is not None :
        log_file = save_dir + log_file
        filehandler = logging.FileHandler(log_file)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)

    logger.setLevel(level)

    return logger


def prepare_device(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    np.set_printoptions(linewidth=np.nan, precision=2)
    torch.set_printoptions(precision=3, linewidth=150)
    if args.device != "cpu":
        # torch.cuda.set_device("cuda:0")
        torch.cuda.init()
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def preprocess(images, args):
    # if images.dtype == torch.float32:
    #     return images.to(args.device)
    # return (images.float()/255).to(args.device) #0.04
    images = images.to(args.device)
    return (images.float() - 127.5)/127.5 # 0.005

def get_env_object(args):
    if args.num_obj == 20:
        return Objects20
    elif args.num_obj == 1:
        return Objects1
    elif args.num_obj == 5:
        return Objects5
    elif args.num_obj == 40:
        return Objects40
    elif args.num_obj == 80:
        return Objects80
    elif args.num_obj == 1000:
        return Objects4k_untex_small
    elif args.num_obj == 4000:
        return Objects4k_untex
    elif args.num_obj == 4001:
        return Objects4k_tex


def build_envname(args, background = None, env_name = None, aperture =None, rotate = True):
    if env_name == "move_play":
        env_name = "full_play"
    env_name = env_name if env_name is not None else args.env_name
    aperture = aperture if aperture is not None else args.aperture

    env_name = env_name+str(args.num_obj)
    env_name = env_name+"_back"+str(args.background if background is None else background)
    env_name = env_name if args.aperture == 0 else env_name + "_app" + str(int(aperture))
    env_name = env_name if args.closeness == 1. else env_name + "_clo" + str(args.closeness)
    env_name = env_name if not args.quality else env_name + "_quality"
    env_name = env_name if not args.foveation else env_name + "_fov"+str(args.foveation)
    env_name = env_name if args.rotate == 1 or args.rotate == 2 or rotate is False else env_name + "_rot"+str(args.rotate)

    env_name = env_name if args.noise == 3 or args.noise == 0 or args.env_name =="full_play" else env_name + "_noi"+str(args.noise)
    env_name = env_name if not args.binocular else env_name + "_binoc"
    env_name = env_name if not args.incline else env_name + "_face"

    return env_name

def get_standard_dataset(args):
    e_name = args.env_name
    if args.env_name in ["tdw_toys", "tdw_toys_cont"]:
        e_name = "tdw_toys"
    if args.env_name in ["full_play", "move_play"]:
        e_name = "full_play"
    env_name = build_envname(args, env_name=e_name, rotate = (args.env_name != "full_play"))
    # if args.env_name == "full_play":
    #     env_name += "_r"
    path_dataset = os.environ["DATASETS_LOCATION"]+env_name+"_dataset"
    return path_dataset

def get_roll_dataset(args):
    e_name = args.env_name
    if args.env_name in ["tdw_toys", "tdw_toys_cont"]:
        e_name = "tdw_toys"
    if args.env_name in ["full_play", "move_play"]:
        e_name = "full_play"
    env_name = build_envname(args, env_name=e_name, rotate=(args.env_name != "full_play"))+"_roll"
    # if args.env_name == "full_play":
    #     env_name += "_r"
    path_dataset = os.environ["DATASETS_LOCATION"] + env_name + "_dataset"
    return path_dataset

def get_test_dataset(args):
    name = None
    if args.env_name in ["full_play", "move_play"]:
        # name = build_envname(args, rotate=False)+"_r_dataset"
        name = build_envname(args, rotate=False, env_name="full_play")+"_dataset"
    elif args.env_name in ["tdw_toys", "tdw_toys_cont"]:
        name = "tdw_room_toys"+str(args.num_obj)+"_back0_app20"+("" if args.noise == 3 or args.noise == 0 else "_noi"+str(args.noise))+"_dataset"
    elif args.env_name == "tdw_room_toys":
        if args.background < 10:
            name = build_envname(args, background=10)+"_dataset"
        else:
            name = build_envname(args, background=0)+"_dataset"
    if name is None:
        Exception("No test dataset for this environment")
    # return os.getcwd()+"/resources/datasets/"+env_name+"_dataset"
    return os.environ["DATASETS_LOCATION"]+name

def get_normal_dataset(args):
    name = None
    if args.env_name in ["full_play", "move_play"]:
        if args.background < 10:
            name = build_envname(args, rotate=False, env_name="full_play")+"_r"
        else:
            name =  build_envname(args,background=10+(args.background)%40,rotate=False, env_name="full_play")
    elif args.env_name in ["tdw_toys", "tdw_toys_cont"]:
        name = "tdw_room_toys"+str(args.num_obj)+"_back10_app20" + ("" if args.noise == 3 or args.noise == 0 else "_noi"+str(args.noise))
    elif args.env_name == "tdw_room_toys":
        # name = "tdw_toys20_back2_app20"
        if args.background < 10:
            name = "tdw_room_toys"+str(args.num_obj)+"_back10_app20"
        else:
            name = "tdw_room_toys"+str(args.num_obj)+"_back0_app20"
    if name is None:
        Exception("No normal dataset for this environment")
    # return os.getcwd()+"/resources/datasets/"+env_name+"_dataset"
    return os.environ["DATASETS_LOCATION"]+name+"_dataset"

def build_default_act_space(args):
    # if args.augmentations =="standard" or args.augmentations == "standard2" or args.augmentations == "standard3":
    #     return [False, True, True, True, True, True, True, args.switch, True, True]
    # return [False]*7
    return [args.def_turn, args.rotate in [0,1], args.noise == 0, args.noise == 0, not args.depth, not args.elevation, not args.focus, args.switch, args.pitch in [0,1], args.roll in [0,1]]




def get_augmentations(args):
    transformations = []
    if args.resize_crop:
        transformations.append(transforms.RandomResizedCrop(size=128,scale=(args.resize_crop,1.0)))
    if args.flip:
        transformations.append(transforms.RandomHorizontalFlip(p=0.5))
    if args.jitter:
        s=1
        transformations.append(transforms.RandomApply([transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8))
    if args.grayscale:
        transformations.append(transforms.RandomGrayscale(p=0.2))
    if args.blur:
        transformations.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=13)],p=0.2))

    return transforms.Compose(transformations)

class TdwImageDataset(Dataset):
    def     __init__(self, args, annotations_file, img_dir="../resources", transform=None, target_transform=None, views =False, new=False,num_views=36, gap=1):
        self.img_labels = pd.read_csv(img_dir+"/"+annotations_file, header=None,sep=",")
        self.args=args
        if new:
            self.img_labels = self.img_labels.loc[self.img_labels.iloc[:,-1] <= "c"]
        self.num_views = num_views
        self.gap=gap
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.num_columns = len(self.img_labels.columns)
        self.totalClasses = 0
        # for base, dirs, files in os.walk(img_dir):
        #     for _ in dirs:
        #         self.totalClasses += 1

        self.back_to_int, i = {}, 0
        for index, _ in self.img_labels.groupby(3).count().iterrows():
            self.back_to_int[index] = i
            i+=1
        self.max_backgrounds = i

        self.obj_to_int, i = {}, 0
        self.list_of_objects = []
        for index, _ in self.img_labels.groupby(2).count().iterrows():
            self.obj_to_int[index] = i
            self.list_of_objects.append(index)
            i+=1
        self.totalClasses = i

        if self.num_columns > 5:
            objects_type = get_env_object(args)
            categories = objects_type.get_categories()
            self.cat_to_int = categories[0]
            self.list_of_categories = categories[1]
            # self.cat_to_int, i = {}, 0
            # self.list_of_categories = []
            # for index, _ in self.img_labels.groupby(5).count().iterrows():
            #     self.cat_to_int[index] = i
            #     self.list_of_categories.append(index)
            #     i+=1


        positions = self.img_labels.groupby(4).count()
        self.max_positions = positions.shape[0]

        self.views = views
        # if self.views:
        #     self.view_number = self.img_labels.iloc[:, 0].split("_")[2]

    def __len__(self):
        # print(len(self.img_labels), self.img_labels.count())
        return len(self.img_labels)

    def get_image(self, img_path):
        if self.args.binocular:
            image_vc = read_image(img_path.replace(".png","_vc.png"))
            image_vd = read_image(img_path.replace(".png","_vd.png"))
            if self.transform:
                image_vc = self.transform(image_vc)
                image_vd = self.transform(image_vd)
            image = torch.cat((image_vc,image_vd),dim=0)
        else:
            image = read_image(img_path)
            if self.transform: image = self.transform(image)
        return image

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 2],self.img_labels.iloc[idx, 0])#+".png"
        image= self.get_image(img_path)
        # image = image.swapaxes(1,2)
        label = self.obj_to_int[self.img_labels.iloc[idx, 2]]
        if self.views:
            return image, label, self.cat_to_int[self.img_labels.iloc[idx, 5]], int(self.img_labels.iloc[idx, 0].split("_")[-3]), self.get_other_image(idx)

        if self.num_columns > 5:
            return image, label, self.back_to_int[self.img_labels.iloc[idx, 3]], self.img_labels.iloc[idx, 4], self.cat_to_int[self.img_labels.iloc[idx, 5]]

        if self.num_columns > 3:
            return image, label, self.back_to_int[self.img_labels.iloc[idx, 3]], self.img_labels.iloc[idx, 4], label
        return image, label, label, label, label

    def get_other_image(self, idx):
        splitted = self.img_labels.iloc[idx, 0].split("_")
        if self.views == 1:
            nv = int(splitted[-3]) + self.gap
            if nv > self.num_views:
                splitted[-3] = str(nv - self.num_views)
            else:
                splitted[-3] = str(nv)
        if self.views == 2:
            splitted[-3]  = str(random.randint(1,self.num_views))

        new = "_".join(splitted)
        path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 2],new)
        return self.get_image(path)

class SpecificImageDataset(Dataset):
    def __init__(self, args, annotations_file, img_dir="../resources", views =0, new=False,num_views=36, gap=1, full_id = False,v=0):
        self.img_labels = pd.read_csv(img_dir+"/"+annotations_file, header=None,sep=",")
        self.args=args
        result = [r.split("_")[-3] for r in self.img_labels[0]]
        self.img_labels[6]=result
        if v:
            self.img_labels = self.img_labels.loc[self.img_labels.iloc[:, 6]==str(v)]
        self.num_views = num_views
        self.gap=gap
        self.img_dir = img_dir
        self.views = views
        self.obj_to_int, i = {}, 0
        self.list_of_objects = []
        self.full_id = full_id

        for index, _ in self.img_labels.groupby(2).count().iterrows():
            self.obj_to_int[index] = i
            self.list_of_objects.append(index)
            i+=1
        self.totalClasses = i

        objects_type = get_env_object(args)
        categories = objects_type.get_categories()
        self.cat_to_int = categories[0]
        self.list_of_categories = categories[1]



    def __len__(self):
        # print(len(self.img_labels), self.img_labels.count())
        return len(self.img_labels)

    def get_image(self, img_path):
        if self.args.binocular:
            image_vc = read_image(img_path.replace(".png","_vc.png"))
            image_vd = read_image(img_path.replace(".png","_vd.png"))
            image = torch.cat((image_vc,image_vd),dim=0)
        else:
            image = read_image(img_path)
        return image

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 2],self.img_labels.iloc[idx, 0])#+".png"
        image = self.get_image(img_path)
        # image = image.swapaxes(1,2)
        if not self.full_id:
            return image, self.obj_to_int[self.img_labels.iloc[idx, 2]], self.cat_to_int[self.img_labels.iloc[idx, 5]], int(self.img_labels.iloc[idx, 0].split("_")[-3]), self.get_other_image(idx)
        return image, self.obj_to_int[self.img_labels.iloc[idx, 2]], self.cat_to_int[self.img_labels.iloc[idx, 5]], int(self.img_labels.iloc[idx, 0].split("_")[-3])#, self.img_labels.iloc[idx, 0]

    def get_other_image(self, idx):
        splitted = self.img_labels.iloc[idx, 0].split("_")
        if self.views == 1:
            nv = int(splitted[-3]) + self.gap
            if nv > self.num_views:
                splitted[-3] = str(nv - self.num_views)
            else:
                splitted[-3] = str(nv)
        if self.views == 2:
            splitted[-3]  = str(random.randint(1,self.num_views))

        new = "_".join(splitted)
        path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 2],new)
        return self.get_image(path)

class TestImageDataset(Dataset):
    def __init__(self, img_dir,views=0):
        self.img_dir = img_dir
        self.views=views

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, "dolphin_002","dolphin_002_10_0_0.png")
        image = read_image(img_path)
        if self.views:
            return image, 1, 0, 10,  read_image(os.path.join(self.img_dir, "dolphin_002","dolphin_002_11_0_0.png"))
        else:
            return image

def get_representations_dataset(args, dataset,network):
    dataloader = DataLoader(dataset, batch_size=len(dataset) + 1, shuffle=False)
    images, labels, labels_background, labels_positions, labels_category = next(iter(dataloader))
    images2 = preprocess(images, args)
    with torch.no_grad():
        if images.shape[0] > 5000:
            features = torch.cat((network(images2[:5000])[0], network(images2[5000:])[0]), dim=0)
        else:
            features, _ = network(images2)
        labels_category = labels_category.to(args.device)
    # _, sol_cat,_,_= lls(features, labels_category, torch.max(labels_category)+1)

    return features, labels_category, images