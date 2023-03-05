import argparse

def str2table(v):
    return list(map(int,v.split(',')))


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--dataset", type=str, default="toybox")
    parser.add_argument("--path", type=str, default="../datasets/ToyBox")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--method", type=str, default="combine2")
    parser.add_argument("--loss", type=str, default="simclr")
    parser.add_argument("--network", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="../gym_results/toybox_results/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--projection", type=int, default=1)
    parser.add_argument("--store_image", type=int, default=0)
    parser.add_argument("--save", type=int, default=1)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--full_load", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--start_epoch", type=int, default=0)

    #SimCLR
    parser.add_argument("--crop", type=float, default=0.5)
    parser.add_argument("--flip", type=str2bool, default=True)
    parser.add_argument("--grayscale", type=str2bool, default=True)
    parser.add_argument("--blur", type=str2bool, default=False)
    parser.add_argument("--pcolor", type=float, default=0.8)

    ##### Toybox
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--remove_hodgepodge", type=int, default=0)
    parser.add_argument("--path2", type=str, default="../datasets/dataset_simclr")


    args = parser.parse_args()
    return args