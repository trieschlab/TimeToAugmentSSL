#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import csv
import os
import time

import matplotlib
import torch
import torchvision
from PIL import Image
from matplotlib.colors import ListedColormap
from sklearn.cluster import AgglomerativeClustering
from torch.linalg import lstsq
import torch.nn.functional as F
import pacmap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.cm import get_cmap
import seaborn as sb
import numpy as np
#
# import config
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, SubsetRandomSampler, BatchSampler
from torchvision.io import write_png

from envs.objects import Objects4k_tex, Objects4k_untex
from tools.logger import EpochLogger
from tools.utils import get_env_object, TdwImageDataset, get_standard_dataset, preprocess, build_envname, \
    get_representations_dataset, SpecificImageDataset, get_augmentations


@torch.no_grad()
def lls(representations, labels, n_classes):
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
def wcss_bcss(args, representations, labels, n_classes, save_dir=None, step_tests=None):
    """
        Calculate the within-class and between-class average distance ratio
        params:
            representations: the representations to be evaluated (Tensor)
            labels: labels of the original data (LongTensor)
        return:
            wb: the within-class and between-class average distance ratio (float)
    """
    representations = torch.stack([representations[labels == i] for i in range(n_classes)])
    centroids = representations.mean(1, keepdim=True)
    wcss = (representations - centroids).norm(dim=-1).mean()
    bcss_nomean = F.pdist(centroids.view(-1,representations.shape[2]))

    OBJ = get_env_object(args)
    if save_dir and centroids.shape[0] == 20:
        dist_matrix = torch.cdist(centroids.squeeze(), centroids.squeeze())
        fig, ax = plt.subplots(1, 1)
        c = ax.pcolor(dist_matrix.cpu().numpy())
        fig.colorbar(c, ax=ax)
        ax.set_title('Similarity matrix')
        plt.yticks(range(centroids.shape[0]), OBJ.get_records_name())
        plt.xticks(range(centroids.shape[0]), OBJ.get_records_name(),rotation=90)
        fig.tight_layout()
        if not os.path.isdir(save_dir + "/similarity/"):
            os.makedirs(save_dir + "/similarity/")
        plt.savefig(save_dir + "/similarity/"+str(step_tests)+".png")
        plt.close()
        #c = plt.imshow(dist_matrix.cpu().numpy())

    bcss = bcss_nomean.mean()
    wb = wcss / bcss
    return wb

@torch.no_grad()
def disentangled_images(representations, *args):
    disentangled_image(representations, *args, indice1=0, indice2=-2)
    disentangled_image(representations, *args, indice1=5, indice2=100)
    disentangled_image(representations, *args, indice1=53, indice2=230)
    disentangled_image(representations, *args, indice1=526, indice2= 624)
    disentangled_image(representations, *args, indice1=701, indice2= 999)
    disentangled_image(representations, *args, indice1=871, indice2= 943)

def plot_importance(sol, sol_b, labs, labs_b, representations, save_file):
    # sol_b_mean = torch.abs(sol_b).mean(dim=1)
    # sol_mean = torch.abs(sol).mean(dim=1)

    # pred_obj = representations @ sol
    sol.requires_grad_()
    representations = representations.detach().clone()
    representations.requires_grad_()
    sol_b.requires_grad_()

    pred_obj = torch.mm(representations, sol)

    # sol.grad.data.mean(dim=1)
    # print(pred_obj.shape, labs.shape, F.cross_entropy(pred_obj, labs).shape)
    F.cross_entropy(pred_obj, labs).backward()
    # print(representations.grad.data.abs().shape)
    sol_mean = representations.grad.data.abs().mean(dim=0)
    representations.grad.data.zero_()

    pred_back = torch.mm(representations, sol_b)
    F.cross_entropy(pred_back, labs_b).backward()
    sol_b_mean = representations.grad.data.abs().mean(dim=0)
    representations.grad.data.zero_()

    # def normalize(v):
    #     return v/torch.max(v)
    # corr_matrix= torch.abs(torch.stack((normalize(sol_mean), normalize(sol_b_mean)), dim=0)).cpu().numpy()
    # color_map = get_cmap('viridis')
    # plt.matshow(corr_matrix,cmap=color_map)

    fig, ax = plt.subplots(1, 1)
    c = ax.pcolor(torch.stack((sol_mean, sol_b_mean), dim=0).cpu().numpy())
    fig.colorbar(c, ax=ax)
    ax.set_title('Feature importance for objects and back recognition')
    from envs.objects import Objects20
    plt.yticks([0,1], [ "objects", "backgrounds"])
    plt.xticks(range(sol_mean.shape[0]))
    fig.tight_layout()
    # if not os.path.isdir(save_dir + "/confusion/"):
    #     os.makedirs(save_dir + "/confusion/")
    # plt.savefig(save_dir + "/confusion/" + str(step_tests) + ".png")
    plt.savefig(save_file)
    plt.close()



@torch.no_grad()
def get_pacmap(representations, labels, epoch, n_classes, class_labels, embedding = None):
    """
        Draw the PacMAP plot
        params:
            representations: the representations to be evaluated (Tensor)
            labels: labels of the original data (LongTensor)
            epoch: epoch (int)
        return:
            fig: the PacMAP plot (matplotlib.figure.Figure)
    """
    # sb.set()
    # sb.set_style("ticks")
    # sb.set_context('paper', font_scale=1.8, rc={'lines.linewidth': 2})
    # color_map = get_cmap('viridis')
    # legend_patches = [Patch(color=color_map(i / n_classes), label=label) for i, label in enumerate(class_labels)]
    # # save the visualization result
    # embedding = pacmap.PaCMAP(n_dims=2)
    # X_transformed = embedding.fit_transform(representations.cpu().numpy(), init="pca")
    # fig, ax = plt.subplots(1, 1)
    # labels = labels.cpu().numpy()
    # plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels, cmap=color_map, s=0.6)
    # # plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels, s=0.6)
    # plt.title("Pacmap visualization of cluster")
    # plt.xticks([]), plt.yticks([])
    # plt.legend(loc='upper left', bbox_to_anchor=(1., 1.), handles=legend_patches, fontsize=13.8)
    # # ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, size=30, weight='medium')
    # plt.xlabel(f'Epoch: {epoch}')
    # return fig
    import seaborn as sns
    # sns.set_style("ticks")
    # sns.set_context('paper', font_scale=1.8, rc={'lines.linewidth': 2})
    color_map = ListedColormap(sns.color_palette('colorblind', n_classes))
    # legend_patches = [Patch(color=color_map(i / n_classes), label=label) for i, label in enumerate(class_labels)]
    legend_patches = [Patch(color=color_map(i), label=label) for i, label in enumerate(class_labels)]
    # save the visualization result
    if embedding is None:
        embedding = pacmap.PaCMAP(n_dims=2)
    X_transformed = embedding.fit_transform(representations.cpu().numpy(), init="pca")
    fig, ax = plt.subplots(1, 1, figsize=(7.7, 4.8))
    labels = labels.cpu().numpy()
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels, cmap=color_map, s=0.6)
    ax.set_title("Pacmap visualization of cluster")
    plt.xticks([]), plt.yticks([])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1., 1.), handles=legend_patches, fontsize=13.8)
    # ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, size=30, weight='medium')
    plt.xlabel(f'Epoch: {epoch}')
    return fig, embedding

def save_image(save_dir,step, obs, args=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if obs.shape[1] > 3:
        filename = "/" + str(step) + "_vc.png"
        filename2 = "/" + str(step) + "_vd.png"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torchvision.transforms.functional.to_pil_image(obs[0,:3].squeeze()).save(save_dir + filename)
        torchvision.transforms.functional.to_pil_image(obs[0,3:6].squeeze()).save(save_dir + filename2)
        return
    if args is not None:
        aug = get_augmentations(args)
        filename = "/aug_" + str(step) + ".png"
        torchvision.transforms.functional.to_pil_image(aug(obs).squeeze()).save(save_dir + filename)
    filename = "/"+str(step)+".png"
    torchvision.transforms.functional.to_pil_image(obs.squeeze()).save(save_dir+filename)

def confusion_matrix(args, save_dir, prediction, labels, nb_classes, step_tests, dataset):
    # confusion_matrix = torch.zeros((nb_classes, nb_classes))
    # t_pred = prediction.argmax(dim=-1)
    # hot_labels = F.one_hot(labels.to(torch.long), nb_classes)
    # hot_pred = F.one_hot(t_pred.to(torch.long), nb_classes)
    # labels_pred = torch.stack([prediction[labels == i] for i in range(nb_classes)])
    t_pred = prediction.argmax(dim=-1)
    hot_pred = F.one_hot(t_pred.to(torch.long), nb_classes)

    array = []
    for i in range(nb_classes):
        selected = hot_pred[labels == i]
        array.append(selected.sum(dim=0)/selected.shape[0])
    conf_matrix = torch.stack(array)
    # labels_pred = torch.concat([hot_pred[labels == i].sum(dim=1)/labels_pred.shape[0] for i in range(nb_classes)], 0)
    # conf_matrix = labels_pred.sum(dim=1)/labels_pred.shape[1]

    # labels_pred = torch.concat([prediction[labels == i] for i in range(nb_classes)], 0)
    # conf_matrix = labels_pred.sum(dim=1)/labels_pred.shape[1]

    n_conf_matrix = conf_matrix.cpu().numpy()
    fig, ax = plt.subplots(1, 1)
    c = ax.pcolor(n_conf_matrix)
    fig.colorbar(c, ax=ax)
    ax.set_title('Confusion matrix')
    if not args.category:
        ticks = dataset.list_of_objects
    else:
        ticks = dataset.list_of_categories
    plt.yticks(range(nb_classes), ticks)
    plt.xticks(range(nb_classes), ticks, rotation=90)
    fig.tight_layout()
    if not os.path.isdir(save_dir + "/confusion/"):
        os.makedirs(save_dir + "/confusion/")
    plt.savefig(save_dir + "/confusion/" + str(step_tests) + ".png")
    plt.close()

def saliency_map(images, idxs, save_dir, rep_function, features):
    if rep_function.new_inputs is None:
        return
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    embeds = rep_function.new_embed.detach()

    for i in idxs:
        # loss = positives - torch.log(torch.sum(torch.exp(torch.cat((positives,negatives),dim=1)), dim=1)).squeeze()
        ground_image = images[i]
        image = images[i:i+1].double().to(embeds.device)
        image.requires_grad_()
        features_out, _ = rep_function.embed(image)
        features_pair = features[i+1:i+2] if i+1 < images.shape[0] else features[i-1:i]
        positives = rep_function.sim_function_simple(features_out, features_pair)
        negatives = rep_function.sim_function(features_out, embeds).squeeze()
        loss = positives - torch.log(torch.sum(torch.exp(torch.cat((positives, negatives), dim=0)), dim=0)).squeeze()

        (-loss).backward()
        saliency, _ = torch.max(image.grad.data.abs(), dim=1)
        saliency = saliency.reshape(128,128)

        # Reshape the image
        ground_image = ground_image.reshape(-1, 128, 128)

        # Visualize the image and the saliency map
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(ground_image.cpu().detach().numpy().transpose(1, 2, 0))
        ax[0].axis('off')
        ax[1].imshow(saliency.cpu(), cmap='hot')
        ax[1].axis('off')
        plt.tight_layout()
        fig.suptitle('The Image and Its Saliency Map')
        plt.savefig(save_dir+str(i))
        matplotlib.pyplot.close()

def feature_vizualization(args, save_dir, labels, sol, rep_function):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    num_labels = torch.max(labels).item()+1
    ground_image = torch.rand((num_labels, 1, 128, 128), device=args.device, requires_grad=False)*2 -1
    ground_image = ground_image.to(args.device).requires_grad_(True)

    sol = sol.to(args.device)
    opt = Adam([ground_image], lr=0.001)
    rep_function.net.eval()
    # ground_image = torch.stack([ground_image]*3, dim=1)
    ground_images = torch.cat((ground_image, ground_image, ground_image), dim=1)

    cls = Objects4k_untex if args.num_obj == 4000 else Objects4k_tex
    _, int_to_categories = cls.get_categories()
    for e in range(5000):
        features_out, _ = rep_function.net.forward(ground_images)
        prediction = features_out @ sol
        to_maximize = torch.norm(prediction - torch.eye(prediction.shape[0], device=args.device).detach(), dim=1).mean(dim=0)
        # to_maximize = torch.norm(prediction - torch.eye(prediction.shape[0], device=args.device).detach())
        to_maximize.backward()
        # to_maximize = features_out.sum()
        # to_maximize = ground_image.sum()
        # to_maximize = prediction[torch.eye(prediction.shape[0], device=args.device, dtype=torch.bool)].sum()#torch.diagonal(prediction).mean(dim=0)
        # (-to_maximize).backward()
        if e %50 == 0:
            print(ground_image[0,0,64,64].item(), to_maximize.item())
        opt.step()
        opt.zero_grad()
        rep_function.net.zero_grad()
        sol.grad = None
        ground_image.data = torch.clip(ground_image.data, -1, 1)
        #
        # with torch.no_grad():
        #     ground_image.data = torch.clip(ground_image.data,-1,1).clone()
        # ground_image.requires_grad_()
        # opt = SGD([ground_image], lr=1000)

    for i in range(num_labels-1):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(ground_images[i].cpu().detach().numpy().transpose(1, 2, 0)*0.5 + 0.5)
        # ax[0].imshow(ground_image[i].cpu().detach().numpy()*0.5 + 0.5)
        ax.axis('off')
        plt.tight_layout()
        fig.suptitle('Feature '+str(int_to_categories[i]))
        plt.savefig(save_dir+str(int_to_categories[i]))
        matplotlib.pyplot.close()

def feature_closest(args, save_dir, network):
    base = TdwImageDataset(args, "dataset.csv", os.environ["DATASETS_LOCATION"] + build_envname(args, rotate=False) + "_rn90_dataset")
    baset = TdwImageDataset(args, "dataset_test.csv", os.environ["DATASETS_LOCATION"] + build_envname(args, rotate=False) + "_rn90_dataset")
    features_train, labels_train, images_train = get_representations_dataset(args, base, network)
    features_test, labels_test, images_test = get_representations_dataset(args, baset, network)
    features = torch.cat((features_train, features_test),dim=0)
    images = torch.cat((images_train, images_test),dim=0)
    # for i in range(num_labels-1):
    for i in range(args.num_latents):
        features_i = features[:,i:i+1]
        valk,ink = torch.topk(features_i, 3, dim=0)
        save_image(save_dir, str(i)+"_0", images[ink[0]])
        save_image(save_dir, str(i)+"_1", images[ink[1]])
        save_image(save_dir, str(i)+"_2", images[ink[2]])



def log_views(buffer, epoch_logger_longtrain):
    masks = [((buffer.views[:buffer.total_size] >= i*10) & (buffer.views[:buffer.total_size] < (i+1)*10)).sum() for i in range(36)]
    for i in range(36):
        epoch_logger_longtrain.log_tabular("angle"+str(i), masks[i].item())


def log_accuracies(save_dir, pred_cat, labels_category, max_labels_cat, name):
    accs = []
    for i in range(max_labels_cat):
        indexes = (labels_category == i)
        accs.append((pred_cat[indexes] == labels_category[indexes]).float().mean().item())
    with open(save_dir + "/"+name+"_accuracies.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(accs)


def log_actions(aggr_actions, actions_logger):
    n_aggr_actions = np.asarray(aggr_actions)
    for i in range(n_aggr_actions.shape[1]):
        actions_logger.log_tabular("a_m"+str(i), np.mean(n_aggr_actions[:, i], axis=0).item())
        actions_logger.log_tabular("a_v"+str(i), np.var(n_aggr_actions[:, i], axis=0).item())
    return []


def log_rewards_stats(rep_function, max_backgrounds, save_dir, sample):
    log_cond_prob = rep_function.loss.log_cond_prob.cpu()
    log_prob = rep_function.loss.log_prob.cpu()
    # if log_prob.shape[0] <= 1: return
    if rep_function.args.method != "simclr": return
    backgrounds = sample["backgrounds"].squeeze().detach().cpu()
    import csv
    with open(save_dir+"/log_prob.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow([log_prob[backgrounds==i].mean().item() for i in range(max_backgrounds)])
    with open(save_dir+"/log_cond_prob.csv", 'a') as f:
        writer = csv.writer(f)
        # tab=[log_cond_prob[backgrounds==i].mean().item() for i in range(max_backgrounds)]
        writer.writerow([log_cond_prob[backgrounds==i].mean().item() for i in range(max_backgrounds)])

def log_replay_categories(buffer, num_categories, num_objects, save_dir):
    # data_obj = buffer.objects[:buffer.total_size]
    # data_cat = buffer.categories[:buffer.total_size]
    with open(save_dir + "/categories_buffer.csv", 'a') as f:
        writer = csv.writer(f)
        # writer.writerow([(data_cat== i).nonzero().shape[0]/buffer.total_size for i in range(num_categories)])
        writer.writerow((buffer.cpt_per_categories/buffer.total_size).tolist())
    with open(save_dir + "/objects_buffer.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow((buffer.cpt_per_objects/buffer.total_size).tolist())
        # writer.writerow([(data_obj == i).nonzero().shape[0]/buffer.total_size for i in range(num_objects)])

#https://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.pdf
def knn_evaluation(train_features, train_labels, test_features, test_labels, n_classes, rep_function, sim_function = None):
    k=20
    i=0
    correct=0
    size_batch = 512
    expanded_train_label = train_labels.view(1,-1).expand(size_batch,-1)
    retrieval_one_hot = torch.zeros(size_batch*k, n_classes,device=train_features.device)
    while i < test_features.shape[0]:
        endi = min(i+size_batch,test_features.shape[0])
        tf = test_features[i:endi]
        distance_matrix = rep_function.sim_function(tf, train_features) if sim_function is None else sim_function(tf,train_features)
        valk, indk = torch.topk(distance_matrix, k, dim=1)
        # valk is batchsize x k
        # retrieval =  torch.gather(expanded_train_label, 1, indk)
        #get the best labels

        #retrieval_one_hot = torch.zeros(K, C)
        #List of labels [1, 4, 9...]
        if tf.shape[0] < size_batch:
            retrieval_one_hot = torch.zeros(tf.shape[0]*k, n_classes, device=train_features.device)
            expanded_train_label = train_labels.view(1, -1).expand(tf.shape[0], -1)

        retrieval =  torch.gather(expanded_train_label, 1, indk)

        # retrieval_one_hot[:,:] = -10000
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


@torch.no_grad()
def get_representations(args, model, iterator, buffer):
    """
    Get all representations of the dataset given the network and the data loader
    params:
        model: the network to be used (torch.nn.Module)
        data_loader: data loader of the dataset (DataLoader)
    return:
        representations: representations output by the network (Tensor)
        labels: labels of the original data (LongTensor)
    """
    model.eval()
    features = []
    labels = []
    # for data_samples, data_labels in data_loader:
    for ind in iterator:
        features.append(model(preprocess(buffer.next_obs[ind],args))[0])
        labels.append(buffer.categories.to(args.device).long())

    features = torch.cat(features, 0)
    labels = torch.cat(labels, 0)
    return features, labels

def get_all_representations(args, model, dataset, proj=False):
    if args.architecture not in ["resnet18","resnet50"]:
        dataloader = DataLoader(dataset, batch_size=len(dataset)+1, shuffle=False)
        images, labels, labels_background, labels_positions, labels_category = next(iter(dataloader))
        with torch.no_grad():
            if images.shape[0] > 5000:
                preprocessed_inputs = preprocess(images[:5000], args)
                preprocessed_inputs2 = preprocess(images[5000:], args)
                features = torch.cat((model(preprocessed_inputs)[0 if not proj else 1], model(preprocessed_inputs2)[0 if not proj else 1]), dim=0)
            else:
                features = model(preprocess(images, args))[0 if not proj else 1]
            labels = labels.to(args.device)
            labels_background = labels_background.to(args.device)
    else:
        with torch.no_grad():
            dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
            iterator = iter(dataloader)
            features = []
            labels_categories_ar = []
            labels_ar = []
            labels_background_ar = []
            labels_position_ar = []
            # for data_samples, data_labels in data_loader:
            for images, labels, labels_background, labels_positions, labels_category in iterator:
                features.append(model(preprocess(images,args))[0 if not proj else 1])
                labels_categories_ar.append(labels_category.to(args.device))
                labels_ar.append(labels.to(args.device))
                labels_background_ar.append(labels_background.to(args.device))
                labels_position_ar.append(labels_positions.to(args.device))

            features = torch.cat(features, 0)
            labels = torch.cat(labels_ar, 0)
            labels_category = torch.cat(labels_categories_ar, 0)
            labels_background = torch.cat(labels_background_ar, 0)
            labels_positions = torch.cat(labels_position_ar, 0)
    return features, labels, labels_background, labels_positions, labels_category, images

class RewLogger:
    def __init__(self, args, info, save_dir, rep_function):
        self.max_objects = info["total_num_objects"]
        self.max_categories = info["total_num_categories"]
        self.max_views = 36
        self.rep_function=rep_function
        self.save_dir = save_dir
        self.args=args

        os.makedirs(self.save_dir+"/objects")
        os.makedirs(self.save_dir+"/categories")
        os.makedirs(self.save_dir+"/views")
        os.makedirs(self.save_dir+"/views2")

        self.objects = {}
        self.categories = {}
        self.views={}
        self.views2={}
        self.init(self.objects, self.max_objects)
        self.init(self.categories, self.max_categories)
        self.init(self.views, self.max_views)
        self.init(self.views2, self.max_views)

    def init(self, dict, max, full=True):
        dict["cpt"] = torch.zeros((max,))
        dict["log_prob"] = torch.zeros((max,))
        dict["cond_log_prob"]= torch.zeros((max,))
        if self.args.full_logs and full:
            dict["lpo_fix"] = torch.zeros((max,))
            dict["lpo_nofix"] = torch.zeros((max,))
            dict["cpt_fix"] = torch.zeros((max,))
            dict["clpo_nofix"] = torch.zeros((max,))
            dict["clpo_fix"] = torch.zeros((max,))
            dict["cpt_nofix"] = torch.zeros((max,))
            dict["negatives_only"] = torch.zeros((max,))
            dict["no_exp_log_prob"] = torch.zeros((max,))
            dict["lower_tmp"] = torch.zeros((max,))
            dict["higher_tmp"] = torch.zeros((max,))
        if self.args.predictor or self.args.inverse:
            dict["loss_a"] = torch.zeros((max,))
            dict["loss_a_fix"] = torch.zeros((max,))
            dict["loss_a_nofix"] = torch.zeros((max,))
        # if self.args.normalizer:
        dict["rew_norm"] = torch.zeros((max,))



    def update_stats(self, sample):
        # if self.rep_function.action_space.__class__.__name__ == "Discrete":
        #     actions = sample["actions"].squeeze().cpu()
        # else:
        #     actions = sample["actions"].cpu()

        # objects = rep_function.loss.objects.cpu()
        objects_hot = F.one_hot( sample["objects"].squeeze().to(torch.long), self.max_objects).float()
        self.aggregate(self.rep_function,sample, objects_hot, self.objects)

        categories_hot = F.one_hot( sample["categories"].squeeze().to(torch.long), self.max_categories).float()
        self.aggregate(self.rep_function, sample, categories_hot, self.categories)
        views_hot = F.one_hot( torch.div(sample["prev_views"],10,rounding_mode='trunc').squeeze().to(torch.long), self.max_views).float()
        self.aggregate(self.rep_function, sample, views_hot, self.views)
        views_hot2 = F.one_hot( torch.div(sample["views"],10,rounding_mode='trunc').squeeze().to(torch.long), self.max_views).float()
        self.aggregate(self.rep_function, sample, views_hot2, self.views2)

    def aggregate(self,rep_function, sample, objects_hot, resume, full=True):
        # print(rep_function.loss.log_cond_prob[0:10].squeeze())
        resume["cpt"] += objects_hot.sum(dim=0).squeeze().cpu()


        cond_log_prob = rep_function.loss.log_cond_prob
        log_prob = rep_function.loss.log_prob

        resume["log_prob"] += (log_prob.view(-1,1)*objects_hot).sum(dim=0).cpu()
        resume["cond_log_prob"] +=(cond_log_prob.view(-1,1)*objects_hot).sum(dim=0).cpu()

        if self.args.full_logs and full:
            m=sample["fixations"]

            ### Fixations
            fix_objects_hot = objects_hot*(m.float().view(-1,1))
            resume["cpt_fix"] += fix_objects_hot.sum(dim=0).squeeze().cpu()
            resume["lpo_fix"] +=(log_prob.view(-1, 1) * fix_objects_hot.float()).sum(dim=0).cpu()
            resume["clpo_fix"]+= (cond_log_prob.view(-1, 1) * fix_objects_hot.float()).sum(dim=0).cpu()

            ### Switchs
            nofix_objects_hot = objects_hot * ((~m).float().view(-1, 1))
            resume["cpt_nofix"] += nofix_objects_hot.sum(dim=0).squeeze().cpu()
            resume["lpo_nofix"] +=  (log_prob.view(-1, 1) * nofix_objects_hot).sum(dim=0).cpu()
            resume["clpo_nofix"] += (cond_log_prob.view(-1, 1) * nofix_objects_hot).sum(dim=0).cpu()

            ### Additionals
            # print(rep_function.loss.no_exp_log_prob.view(-1,1).shape, objects_hot.sum(dim=0).shape, objects_hot.shape, self.no_exp_log_prob.shape)
            if rep_function.loss.negatives_only is not None:
                resume["negatives_only"] += (rep_function.loss.negatives_only.view(-1,1)*objects_hot).sum(dim=0).cpu()

                resume["no_exp_log_prob"] += (rep_function.loss.no_exp_log_prob.view(-1,1)*objects_hot).sum(dim=0).cpu()
                resume["lower_tmp"] += (rep_function.loss.lower_tmp.view(-1,1) * objects_hot).sum(dim=0).cpu()
                resume["higher_tmp"] += (rep_function.loss.higher_tmp.view(-1,1) * objects_hot).sum(dim=0).cpu()
            if (self.args.predictor > 1 or self.args.inverse) and rep_function.loss.loss_a is not None:
                resume["loss_a"] += (rep_function.loss.loss_a.view(-1, 1) * objects_hot).sum(dim=0).cpu()
                resume["loss_a_fix"] += (rep_function.loss.loss_a.view(-1, 1) * fix_objects_hot).sum(dim=0).cpu()
                resume["loss_a_nofix"] += (rep_function.loss.loss_a.view(-1, 1) * nofix_objects_hot).sum(dim=0).cpu()
            if rep_function.loss.feedback is not None:
                resume["rew_norm"] += (rep_function.loss.feedback.view(-1, 1) * objects_hot).sum(dim=0).cpu()

    def log(self, replay):
        with torch.no_grad():
            # dataset_test = TdwImageDataset(self.args,"dataset_test.csv", get_standard_dataset(self.args))
            # dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test) + 1, shuffle=False)
            # images_test, labels_test, labels_background_test, labels_positions_test, labels_category_test = next(iter(dataloader_test))
            # images_test = preprocess(images_test, args)
            # features_test, _ = self.rep_function.embed(images_test.to(self.args.device))
            features_test, labels_test, labels_background_test, labels_positions_test, labels_category_test, _= get_all_representations(self.args, self.rep_function.net, dataset_test)[0]

            indices = next(iter(BatchSampler(SubsetRandomSampler(range(replay.total_size)), min(replay.total_size,self.args.batch_size), drop_last=True)))
            features, _ = self.rep_function.embed(replay.next_obs[indices].to(self.args.device))

            test_r = -torch.logsumexp(self.rep_function.sim_function(features_test, features )/self.args.temperature,dim=1).view(-1,1).cpu()#.mean(dim=0).detach().cpu().item()
            objects_hot = F.one_hot(labels_test.squeeze().cpu().to(torch.long), torch.max(labels_category_test).item()+1).float()
            categories_hot = F.one_hot(labels_category_test.squeeze().cpu().to(torch.long), self.max_categories).float()
            all_objects_hot = objects_hot.sum(dim=0)
            all_categories_hot = categories_hot.sum(dim=0)
            obj_test_r = (test_r*objects_hot).sum(dim=0)/all_objects_hot
            cat_test_r = (test_r*categories_hot).sum(dim=0)/all_categories_hot
            with open(self.save_dir+"/novel_object.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow(obj_test_r.numpy())
            with open(self.save_dir+"/novel_category.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow(cat_test_r.numpy())

        self.log_dict(self.save_dir+"/objects", self.objects)
        self.log_dict(self.save_dir+"/categories", self.categories)
        self.log_dict(self.save_dir+"/views", self.views)
        self.log_dict(self.save_dir+"/views2", self.views2)

        self.init(self.objects, self.max_objects)
        self.init(self.categories, self.max_categories)
        self.init(self.views, self.max_views)
        self.init(self.views2, self.max_views)

    def log_dict(self, save_dir, resume, full=True):
        with open(os.path.join(save_dir,"log_prob.csv"), 'a') as f:
            writer = csv.writer(f)
            writer.writerow((resume["log_prob"]/(resume["cpt"] +0.00001)).numpy())
        with open(os.path.join(save_dir,"log_cond_prob.csv"), 'a') as f:
            writer = csv.writer(f)
            writer.writerow((resume["cond_log_prob"] / (resume["cpt"] +0.00001)).numpy())
            # writer.writerow(self.rep_function.loss.log_prob).numpy())
        if self.args.full_logs and full :
            with open(save_dir + "/lpo_0.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow((resume["lpo_fix"] / (resume["cpt_fix"] + 0.00001)).numpy())
            with open(save_dir + "/clpo_0.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow((resume["clpo_fix"]/ (resume["cpt_fix"] + 0.00001)).numpy())
            with open(save_dir + "/lpo_2.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow((resume["lpo_nofix"] / (resume["cpt_nofix"] + 0.00001)).numpy())
            with open(save_dir + "/clpo_2.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow((resume["clpo_nofix"] / (resume["cpt_nofix"] + 0.00001)).numpy())
            with open(save_dir + "/cpt_nofix.csv", 'a') as f:
                csv.writer(f).writerow(resume["cpt_nofix"].numpy())
            with open(save_dir + "/cpt_fix.csv", 'a') as f:
                csv.writer(f).writerow(resume["cpt_fix"].numpy())
            if self.rep_function.loss.negatives_only is not None:
                with open(save_dir + "/neg_only.csv", 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow((resume["negatives_only"] / (resume["cpt"] + 0.00001)).numpy())
                with open(save_dir + "/noexp_logprob.csv", 'a') as f:
                    csv.writer(f).writerow((resume["no_exp_log_prob"]/ (resume["cpt"] + 0.00001)).numpy())
                with open(save_dir + "/lowtmp.csv", 'a') as f:
                    csv.writer(f).writerow((resume["lower_tmp"]/ (resume["cpt"]  + 0.00001)).numpy())
                with open(save_dir + "/hightmp.csv", 'a') as f:
                    csv.writer(f).writerow((resume["higher_tmp"]/ (resume["cpt"] + 0.00001)).numpy())
            if self.args.predictor or self.args.inverse:
                with open(save_dir + "/loss_a.csv", 'a') as f:
                    csv.writer(f).writerow((resume["loss_a"]/ (resume["cpt"]  + 0.00001)).numpy())
                with open(save_dir + "/loss_a_fix.csv", 'a') as f:
                    csv.writer(f).writerow((resume["loss_a"]/ (resume["cpt_fix"]  + 0.00001)).numpy())
                with open(save_dir + "/loss_a_nofix.csv", 'a') as f:
                    csv.writer(f).writerow((resume["loss_a"]/ (resume["cpt_nofix"]  + 0.00001)).numpy())
            if self.rep_function.loss.feedback is not None:
                with open(save_dir + "/feedback.csv", 'a') as f:
                    csv.writer(f).writerow((resume["rew_norm"] / (resume["cpt"] + 0.00001)).numpy())

    # _____________________________________________________________________________


def get_first_cluster(model):
    n_samples = len(model.labels_)
    samples_cluster = np.zeros(n_samples,dtype=np.int64)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                samples_cluster[child_idx] = i
    return samples_cluster

def eval_aggl(args, embed_network, test=False, proj =False):
    if args.num_obj == 4000:
        dataset_name = os.environ["DATASETS_LOCATION"]+"full_play4000_back5_app5_clo0.8_dataset"
    else:
        dataset_name = os.environ["DATASETS_LOCATION"]+"full_play4001_back10_app5_clo0.8_dataset"

    dataset = SpecificImageDataset(args, "dataset.csv" if not test else "dataset_test.csv",dataset_name,views=1, full_id=True)
    dataloader = DataLoader(dataset, batch_size=len(dataset) + 1, shuffle=False)
    images, labels, labels_category, labels_views = next(iter(dataloader))
    indices = (labels_views <= 2)
    img_ind, obj_ind, cat_ind, views_ind = images[indices], labels[indices], labels_category[indices], labels_views[indices]
    with torch.no_grad():
        images = preprocess(img_ind, args)
        features = embed_network(images.to(args.device))[0 if not proj else 1].cpu().numpy()

    # import scipy.cluster.hierarchy as sch
    t = time.time()
    model = AgglomerativeClustering(n_clusters=torch.max(cat_ind).item() +1, distance_threshold=None, affinity='euclidean' if args.sim_func != "cosine" else "cosine",linkage='average')
    # model = AgglomerativeClustering(n_clusters=None, distance_threshold=0, affinity='euclidian', linkage='average')
    clusters = model.fit_predict(features)
    clusters_first = get_first_cluster(model)
    from sklearn.metrics import v_measure_score
    v1_s = v_measure_score(cat_ind.numpy(), clusters)
    v2_s = v_measure_score(obj_ind.numpy(), clusters_first)
    return v1_s, v2_s