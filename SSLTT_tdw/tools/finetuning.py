import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from tools.logger import EpochLogger
from torch import nn
from torch.nn import functional as F

# Taken in https://github.com/p3i0t/SimCLR-CIFAR10/blob/master/simclr_lin.py
from tools.utils import preprocess, get_augmentations


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def run_epoch(args, dataloader, augmentation_set, network, lin_model, optimizer=None, scheduler=None):
    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    for i_batch, sample_batched in enumerate(dataloader):
        i1, labels, _, _, labels_category = sample_batched
        labels_category = labels_category.to(args.device).detach()
        # if optimizer is not None and args.method in ["simclr","combine","combine2"]:
        #     i1 = augmentation_set(i1)
        #     imgs = preprocess(i1, args)
        # else:
        imgs = preprocess(i1, args)
        with torch.no_grad():
            embeds = network(imgs)[0]
        logits = lin_model(embeds)
        loss = F.cross_entropy(logits, labels_category)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        acc = (logits.argmax(dim=1) == labels_category).float().mean()
        loss_meter.update(loss.item(), i1.size(0))
        acc_meter.update(acc.item(), i1.size(0))
    return loss_meter.avg, acc_meter.avg

def finetune(args, train_set, test_set, network, save_dir):
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Prepare model
    lin_model = nn.Linear(network.repr, train_set.totalClasses).to(args.device)

    optimizer = torch.optim.Adam(lin_model.parameters(), lr=0.1, weight_decay=args.weight_decay)
    num_epochs = 200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(train_dataloader),1e-4)

    logger = EpochLogger(output_dir=save_dir, exp_name="Seed-" + str(args.seed), output_fname='linear_progress.txt')
    augmentation_set = get_augmentations(args)
    network.eval()
    for epoch in range(1, num_epochs+1):
        lin_model.train()
        train_loss, train_acc = run_epoch(args, train_dataloader, augmentation_set, network, lin_model, optimizer, scheduler)
        lin_model.eval()
        loss, acc = run_epoch(args, test_dataloader, augmentation_set, network, lin_model)
        logger.log_tabular("test_acc", acc)
        logger.log_tabular("loss", loss)
        logger.log_tabular("train_acc", train_acc)
        logger.log_tabular("train_loss", train_loss)
        logger.dump_tabular()
    network.train()




if __name__ == '__main__':
    finetune()