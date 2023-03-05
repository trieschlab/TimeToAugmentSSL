import torch
from torch.nn import functional as F


def apply(argss, *args, **kwargs):
    if argss.loss == "simclr":
        loss_met = apply_simclr(argss,  *args, **kwargs)
    if argss.loss == "byol":
        loss_met = apply_byol(argss,  *args, **kwargs)
    if argss.loss == "vicreg":
        loss_met = apply_vicreg(argss,  *args, **kwargs)
    return loss_met

def prepare_inputs(argss, proj):
    return proj

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def apply_vicreg(argss, proj, *args, **kwargs):
    proj = prepare_inputs(argss,proj)
    x, y = proj.split(proj.shape[0] //2)

    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    repr_loss = F.mse_loss(x, y)
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

    cov_x = (x.T @ x) / (argss.batch_size - 1)
    cov_y = (y.T @ y) / (argss.batch_size - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(128) + off_diagonal(cov_y).pow_(2).sum().div(128)
    return cov_loss + std_loss + repr_loss


def apply_simclr(argss, proj, network, mask, *args, **kwargs):
    proj = prepare_inputs(argss,proj)
    e1, e2 = proj.split(proj.shape[0] //2)
    pos_loss = F.cosine_similarity(e1, e2, dim=1) / argss.temperature
    sim_matrix = F.cosine_similarity(proj.unsqueeze(1), proj.unsqueeze(0), dim=2) / argss.temperature
    sim_matrix = sim_matrix[mask].reshape(e1.shape[0], -1)
    neg_loss = -torch.logsumexp(sim_matrix, dim=1)
    return -pos_loss - neg_loss

def apply_byol(argss, proj, network, mask, predictor, target_network, all_imgs,*args, **kwargs):
    # imgs1, imgs2 = all_imgs.split(all_imgs.shape[0] //2)
    proj = prepare_inputs(argss,proj)
    x = predictor(proj)
    e_target, x_target = target_network(all_imgs)
    x_target = x_target.detach()
    x_mix = torch.cat((x[:x.shape[0] // 2], x_target[:x.shape[0] // 2]), dim=0)
    y_mix = torch.cat((x_target[x.shape[0] // 2:], x[x.shape[0] // 2:]), dim=0)

    x_mix = F.normalize(x_mix, dim=-1, p=2)
    y_mix = F.normalize(y_mix, dim=-1, p=2)
    return 2 - 2 * F.cosine_similarity(x_mix, y_mix, dim=1) #/ args.temperature
