import os
import random
import time

import pandas as pd
import torch
import torchvision
from torch.linalg import lstsq
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.io import read_image
from torch.nn import functional as F
import numpy as np

from models import ResNet18, MLP


def prepare_device(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    np.set_printoptions(linewidth=np.nan, precision=2)
    torch.set_printoptions(precision=3, linewidth=150)
    if args.device != "cpu":
        # torch.cuda.set_device("cuda:0")
        # torch.cuda.init()
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def preprocess_all(i1,i2,args, augmentation_set):
    img = preprocess(i1, args)
    if args.method == "time":
        img2 = preprocess(i2, args)
    elif args.method == "simclr":
        i2 = augmentation_set(i1)
        img2 = preprocess(i2, args)
    elif args.method == "combine":
        i2 = augmentation_set(i2)
        img2 = preprocess(i2, args)
    else:
        img2 = preprocess(i2, args)
    return img, img2

def preprocess(images, args):
    images = images.to(args.device)
    return (images.float() - 127.5)/127.5

def get_dataset(args):
    train_dataset = ImageDataset(args, "dataset.csv", os.path.abspath(args.path))
    val_dataset = ImageDataset(args, "dataset.csv", os.path.abspath(args.path), pair=False)
    test_dataset = ImageDataset(args, "test_dataset.csv", os.path.abspath(args.path), pair=False)
    return train_dataset, val_dataset, test_dataset


def get_augmentations(args):
    transformations = []
    if args.crop != 1:
        transformations.append(transforms.RandomResizedCrop(size=(162,288), scale=(args.crop, 1.0)))#, scale=(0.08, 1.0)
    if args.flip:
        transformations.append(transforms.RandomHorizontalFlip(p=0.5))
    if args.pcolor != 0:
        s=1
        transformations.append(transforms.RandomApply([transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=args.pcolor))
    if args.grayscale:
        transformations.append(transforms.RandomGrayscale(p=0.2))
    if args.blur:
        transformations.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=13)],p=0.2))

    return transforms.Compose(transformations)


def get_networks(args):
    if args.network == "resnet18":
        network = ResNet18(args).to(args.device)
    elif args.network == "convnet":
        network = ConvNet(args, 128).to(args.device)
    return network


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def save_image(save_dir,step, obs):
    filename = "/"+str(step)+".png"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torchvision.transforms.functional.to_pil_image(obs).save(save_dir+filename)

@torch.no_grad()
def lls(representations, labels, n_classes, solution = None):
    if solution is None:
        one_hots_long = F.one_hot(labels, n_classes)
        one_hots = one_hots_long.to(torch.float32)
        ls = lstsq(representations, one_hots)
        solution = ls.solution
    prediction = representations @ solution

    # print("Original performance:",torch.norm(prediction - one_hots.detach(), dim=1).mean(dim=0))

    hard_prediction = prediction.argmax(dim=-1)
    acc = (hard_prediction == labels).sum() / len(representations)
    five_predictions = torch.topk(prediction, min(5,prediction.shape[1]), dim=-1).indices
    acc5 = (labels.unsqueeze(-1) == five_predictions).sum() / len(representations)
    return acc, solution, prediction, acc5


@torch.no_grad()
def get_representations(args, model, data_loader, epoch):
    """
    Get all representations of the dataset given the network and the data loader
    params:
        model: the network to be used (torch.nn.Module)
        data_loader: data loader of the dataset (DataLoader)
    return:
        representations: representations output by the network (Tensor)
        labels: labels of the original data (LongTensor)
    """
    features = []
    labels = []
    i=0
    for data_images, data_labels in data_loader:
        i+=1
        images = preprocess(data_images, args)
        features.append(model(images)[0])
        labels.append(data_labels.to(args.device))
        if epoch < 40 and i >= 10:
            break


    features = torch.cat(features, 0)
    labels = torch.cat(labels, 0)
    return features, labels

@torch.no_grad()
def linear_evaluation(args,network, train_dataset, test_dataset, logger, epoch):
    network.eval()
    test_time = time.time()

    train_dataloader = DataLoader(train_dataset, batch_size=min(1024,len(train_dataset) + 1), shuffle=False)
    features, labels = get_representations(args, network, train_dataloader, epoch)
    max_labels = (torch.max(labels)+1).item()
    acc, solution, prediction, acc5 = lls(features, labels, max_labels)

    test_dataloader = DataLoader(test_dataset, batch_size=min(1024,len(test_dataset) + 1), shuffle=False)
    test_features, test_labels = get_representations(args, network, test_dataloader, epoch)

    acc_test, _ , prediction_test, acc5_test = lls(test_features, test_labels, max_labels, solution=solution)
    acc_val, _ , prediction_val, acc5_val = lls(test_features, test_labels, max_labels)

    # acc_knn =  knn_evaluation(args,features, labels, test_features, test_labels, max_labels)
    logger.log_tabular("train_acc", acc.item())
    logger.log_tabular("train_acc5", acc5.item())
    # logger.log_tabular("acc_cat_knn", acc_knn.item())
    logger.log_tabular("test_acc", acc_test.item())
    logger.log_tabular("test_acc5", acc5_test.item())
    logger.log_tabular("val_acc", acc_val.item())
    logger.log_tabular("val_acc5", acc5_val.item())
    logger.log_tabular("epoch", epoch)
    logger.log_tabular("time", time.time()-test_time)
    logger.dump_tabular()
    network.train()

def knn_evaluation(args, train_features, train_labels, test_features, test_labels, n_classes):
    k=20
    i=0
    correct=0
    size_batch = 512
    expanded_train_label = train_labels.view(1,-1).expand(size_batch,-1)
    retrieval_one_hot = torch.zeros(size_batch*k, n_classes,device=train_features.device)
    while i < test_features.shape[0]:
        endi = min(i+size_batch,test_features.shape[0])
        tf = test_features[i:endi]
        distance_matrix = F.cosine_similarity(tf,  train_features, dim=2)/args.temperature
        # rep_function.sim_function(tf, train_features)
        valk, indk = torch.topk(distance_matrix, k, dim=1)
        if tf.shape[0] < size_batch:
            retrieval_one_hot = torch.zeros(tf.shape[0]*k, n_classes, device=train_features.device)
            expanded_train_label = train_labels.view(1, -1).expand(tf.shape[0], -1)

        retrieval =  torch.gather(expanded_train_label, 1, indk)
        rt_onehot = retrieval_one_hot.scatter(1, retrieval.view(-1,1) , 1)
        rt_onehot = rt_onehot.view(retrieval.shape[0],k, n_classes)
        not_available = (rt_onehot.sum(dim=1) ==0)
        sim_topk = rt_onehot*valk.unsqueeze(-1)

        probs = torch.sum(sim_topk, dim=1)
        probs[not_available] = -10000
        prediction = torch.max(probs,dim=1).indices
        correct += (prediction == test_labels[i:endi]).sum(dim=0)
        i=endi

    return correct/test_features.shape[0]



class ImageDataset(Dataset):
    def __init__(self, args, annotations_file, img_dir="../resources",pair=True):
        self.img_labels = pd.read_csv(os.path.join(img_dir,annotations_file), header=None,sep=",")
        if args.remove_hodgepodge:
            self.img_labels = self.img_labels.loc[self.img_labels[4] != "hodgepodge"]
        self.img_dir = img_dir
        self.args =args
        self.pair = pair
        #groups = self.img_labels.groupby(2)
        self.selection = self.img_labels.loc[self.img_labels[5] == 1]
        self.cat_to_int, i = {}, 0
        self.list_of_categories = []
        for index, _ in self.img_labels.groupby(1).count().iterrows():
            self.cat_to_int[index] = i
            self.list_of_categories.append(index)
            i += 1

        self.transformed_img = False
        if args.method == "combine2" or args.method == "simclr2":
            self.transformed_img = True



    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        if not self.pair:
            return image, self.cat_to_int[self.img_labels.iloc[idx, 1]]
        if self.args.method in ["simclr"]:
            # image, self.cat_to_int[self.img_labels.iloc[idx, 1]], image,t
            img_pair = image
        else:
            img_pair = self.get_other_image( img_path, self.img_labels.iloc[idx, 5])
        return image, self.cat_to_int[self.img_labels.iloc[idx, 1]], img_pair

    def get_other_image(self, img_path, view):
        if self.transformed_img:
            img_path = img_path.replace(self.args.path, self.args.path2)
        if self.args.method == "simclr2":
            return read_image(img_path)
        cut = img_path.split("/")
        next_view =  "%04d" % (1+int(view))

        cut[-1] = cut[-1].replace("%04d" %view,next_view)
        new_file = "/".join(cut)
        isfile = os.path.isfile(new_file)

        if not isfile:
            img_path2 = self.selection.iloc[random.randint(0, len(self.selection) - 1), 0]
            return read_image(img_path2)
        else:
            prev_view = next_view
            speed = random.randint(1, self.args.speed)
            if speed == 1:
                return read_image(new_file)
            next_view = "%04d" % (speed + int(view))
            cut[-1] = cut[-1].replace(prev_view, next_view)
            new_file = "/".join(cut)
            isfile = os.path.isfile(new_file)
            prev_view = next_view
            i=1
            while not isfile:
                next_view = "%04d" % (speed + int(view) -i)
                cut[-1] = cut[-1].replace(prev_view, next_view)
                new_file = "/".join(cut)
                isfile = os.path.isfile(new_file)
                prev_view = next_view
                i+=1
            return read_image(new_file)

