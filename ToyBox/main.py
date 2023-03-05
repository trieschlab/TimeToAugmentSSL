import argparse
import datetime
import os
import time

import torch
from torch.utils.data import DataLoader

from arguments import get_args
from logger import EpochLogger
from losses import *
from models import ResNet18, MLPHead, save
from utils import *

args = get_args()
os.path.abspath(args.path)
prepare_device(args)
save_dir = args.save_dir + datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + '_' + str(args.seed)
os.makedirs(save_dir)

epoch_logger = EpochLogger(output_dir=save_dir, exp_name="Seed-" + str(args.seed), output_fname='progress.txt')
epoch_logger.save_config(args)

augmentation_set = get_augmentations(args)
network= get_networks(args)
parameters = list(network.parameters())

predictor, target_network = None, None
if args.loss == 'byol':
    predictor = MLPHead(128,256,128).to(args.device)
    parameters = parameters + list(predictor.parameters())
    target_network = ResNet18(args).to(args.device)
    soft_update(target_network, network, tau=1)
    for param_target in target_network.parameters():
        param_target.requires_grad = False  # not update by gradient

optimizer = torch.optim.AdamW(parameters, lr=5e-4, weight_decay=0.01)


if args.load != "":
    print("loading")
    path_load = args.load + "model.pt"
    checkpoint = torch.load(path_load, map_location=torch.device(args.device))
    network.load_state_dict(checkpoint['network_state_dict'])
    optimizer.load_state_dict(checkpoint['network_optimizer_state_dict'])
network.train()


mask = torch.ones(2*args.batch_size, 2*args.batch_size, dtype=torch.bool)
mask = mask.fill_diagonal_(0)

train_dataset, val_dataset, test_dataset = get_dataset(args)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)



t=time.time()
all_time = t
all_loss_met = torch.tensor([0.],device=args.device)
i_batch = 0
for e in range(args.start_epoch,args.num_epochs):
    if e%10 == 0:
        if e > 0:save(args, save_dir, network, optimizer)
        epoch_logger.log_tabular("train_time", time.time()-t)
        epoch_logger.log_tabular("loss_inv", all_loss_met.cpu().item()/(i_batch+1))
        t = time.time()
        linear_evaluation(args, network, test_dataset, val_dataset, epoch_logger, e)
    print("Start epoch", e)
    all_loss_met = torch.tensor([0.], device=args.device)

    for i_batch, sample_batched in enumerate(train_dataloader):
        i1, label, i2 = sample_batched
        img, img2 = preprocess_all(i1,i2,args,augmentation_set)
        if e == 0 and i_batch < 10:
            save_image(save_dir, i_batch, i1[0])
            save_image(save_dir, i_batch+1000, i2[0])

        all_imgs = torch.cat((img,img2),dim=0)
        embeds, proj = network(all_imgs)
        loss_met = apply(args, proj, network, mask, predictor, target_network, all_imgs).mean(dim=0)
        all_loss_met += loss_met.detach()
        loss_met.backward()
        optimizer.step()
        optimizer.zero_grad()
        if args.loss == "byol":
            soft_update(target_network, network, 0.005)

    # if time.time() - all_time > 65000:
save(args, save_dir, network,optimizer)
epoch_logger.log_tabular("train_time", time.time()-t)
epoch_logger.log_tabular("loss_inv", all_loss_met.cpu().item()/(1+i_batch))
linear_evaluation(args, network, test_dataset, val_dataset, epoch_logger, args.num_epochs)





