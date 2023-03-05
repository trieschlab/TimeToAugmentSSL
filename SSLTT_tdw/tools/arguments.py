import argparse
import json


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2table(v):
    return v.split(',')


def parse_datasets():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name",type=str,default="full_play")
    parser.add_argument("--display",type=str,default=":1.0")
    parser.add_argument("--num_obj",type=int, default=4000)
    parser.add_argument("--binocular",type=str2bool, default=False)
    parser.add_argument("--aperture",type=float, default=5)
    parser.add_argument("--closeness",type=float, default=1.)
    parser.add_argument("--num_actions",type=int, default=3)
    parser.add_argument("--obj_mode",type=int, default=0)
    parser.add_argument("--background",type=int, default=5)
    parser.add_argument("--noise",type=float, default=0)#3
    parser.add_argument("--rotate",type=int, default=2)
    parser.add_argument("--quality",type=str2bool, default=False)
    parser.add_argument("--switch",type=str2bool, default=False)
    parser.add_argument("--foveation",type=int, default=0)
    parser.add_argument("--random_orient",type=str2bool, default=True)
    parser.add_argument("--min_angle_speed",type=int, default=0)
    parser.add_argument("--max_angle_speed",type=int, default=30)
    parser.add_argument("--cluster_mode",type=str2bool, default=False)
    parser.add_argument("--batch_mode",type=str2bool, default=False)
    parser.add_argument("--local",type=str2bool, default=True)

    parser.add_argument("--depth",type=float,default=0)
    parser.add_argument("--min_depth",type=float,default=0.65)
    parser.add_argument("--elevation",type=float,default=0)
    parser.add_argument("--focus",type=float,default=1.4)
    parser.add_argument("--categories_removal",type=int,default=0)
    parser.add_argument("--room_categories",type=str2bool, default=False)
    parser.add_argument("--incline",type=str2bool, default=False)

    return parser

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name",type=str,default="full_play")
    parser.add_argument("--exp_name",type=str,default="random_name")
    parser.add_argument("--deterministic_eval",type=str2bool,default=True)
    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--log_interval",type=int, default=4000)
    parser.add_argument("--num_updates",type=int,default=8000)
    parser.add_argument("--num_epochs",type=int,default=65)
    parser.add_argument("--start_epochs",type=int,default=0)
    parser.add_argument("--minimum_time",type=int,default=3000)
    parser.add_argument("--epochs_number",type=int,default=0)
    parser.add_argument("--epochs_type",type=int,default=0)
    parser.add_argument("--epochs_length",type=int,default=150)
    parser.add_argument("--separate",type=str2bool,default=False)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--save",default="../gym_results/tests/")
    parser.add_argument("--multisave",type=str2bool, default=False)
    parser.add_argument("--display",default=":1.0")
    parser.add_argument("--log_backgrounds_probs",type=str2bool, default=True)
    parser.add_argument("--full_logs",type=str2bool, default=True)
    parser.add_argument("--back_logs",type=str2bool, default=False)
    parser.add_argument("--knn_logs",type=str2bool, default=True)
    parser.add_argument("--rot_logs",type=str2bool, default=True)

    parser.add_argument("--local",type=str2bool, default=True)
    parser.add_argument("--reset",type=int, default=100)
    parser.add_argument("--reset_correction",type=str2bool, default=True)
    parser.add_argument("--pacmap",type=str2bool, default=True)
    parser.add_argument("--num_obj",type=int, default=4000)
    parser.add_argument("--split_obj",type=int, default=0)

    parser.add_argument("--category",type=str2bool, default=True)
    parser.add_argument("--categories_removal",type=int,default=0)

    parser.add_argument("--train_per_step",type=int, default=1)
    parser.add_argument("--nlearn",type=int,default=1)
    parser.add_argument("--reset_switch",type=str2bool,default=False)




    #Loading and saving
    parser.add_argument("--load_params",type=str,default=None)
    parser.add_argument("--load_model",type=str,default="null")
    parser.add_argument("--save_model",type=str,default="null")
    parser.add_argument("--save_buffer",type=str,default="null")
    parser.add_argument("--load_buffer",type=str,default="null")
    parser.add_argument("--video",type=int,default=0)

    ###DRL Algorithm
    parser.add_argument("--agent",type=str,default="ord_10")
    parser.add_argument("--nord",type=int,default=10)
    parser.add_argument("--double",type=str2bool,default=False)
    parser.add_argument("--gamma",type=float,default=0.9)
    parser.add_argument("--buffer_size",type=int,default=100000)
    parser.add_argument("--batch_size",type=int,default=256)
    parser.add_argument("--tau",type=float,default=1e-2)
    parser.add_argument("--alpha",type=float,default=1.0)
    parser.add_argument("--lr",type=float,default=5e-4)
    parser.add_argument("--neurons",type=int,default=256)
    parser.add_argument("--layers",type=int,default=2)
    parser.add_argument("--double_critic",type=str2bool,default=False)
    parser.add_argument("--reward_type",type=int,default=5)
    parser.add_argument("--imagine",type=str2bool,default=False)
    parser.add_argument("--coef_entropy",type=float,default=1)
    parser.add_argument("--drl_inputs",type=str,default="inputs")#target, ones, inputs, rewards, mix


    #Neural network representation
    parser.add_argument("--lr_rep",type=float,default=5e-4)
    parser.add_argument("--linear_repr",type=str2bool,default=True)
    parser.add_argument("--neurons_rep",type=int,default=256)
    parser.add_argument("--layers_rep",type=int,default=2)
    parser.add_argument("--tau_target_rep",type=float,default=0.01)
    parser.add_argument("--temperature",type=float,default=0.1)
    parser.add_argument("--temperature2",type=float,default=0.1)

    parser.add_argument("--num_latents",type=int, default=128)
    parser.add_argument("--weight_decay",type=float, default=1e-6)
    parser.add_argument("--weight_decay_pred",type=float, default=1e-6)
    parser.add_argument("--number_negatives",type=int, default=0)
    parser.add_argument("--action_negatives",type=int, default=1)
    parser.add_argument("--negatives",type=str2bool, default=True)

    parser.add_argument("--method",type=str,default="simclr")
    parser.add_argument("--augmentations",type=str,default="time")
    parser.add_argument("--regularizers",type=str2table,default="")
    parser.add_argument("--sim_func",type=str,default="cosine")
    parser.add_argument("--projection",type=str2bool,default=False)
    parser.add_argument("--proj_layers",type=int,default=1)
    parser.add_argument("--proj_dropout",type=float,default=0)
    parser.add_argument("--proj_activation",type=str,default="relu")

    parser.add_argument("--temporal_consistency",type=float,default=0)
    parser.add_argument("--architecture",type=str,default="convnet_3")
    parser.add_argument("--importance_sampling",type=str2bool,default=False)
    parser.add_argument("--clip",type=float,default=10)
    parser.add_argument("--queue_size", type=int,default=0)
    parser.add_argument("--predictor",type=int,default=0)
    parser.add_argument("--inverse",type=int,default=0)
    parser.add_argument("--depth_apart",type=str2bool,default=False)
    parser.add_argument("--noise_a",type=float,default=0)
    parser.add_argument("--split",type=str2bool,default=True)
    parser.add_argument("--pred_batchnorm",type=str2bool,default=False)
    parser.add_argument("--pre_norm",type=int,default=0)
    parser.add_argument("--multiforward",type=str2bool,default=False)
    parser.add_argument("--pred_dropout",type=float,default=0)
    parser.add_argument("--ext_dropout",type=str2bool,default=True)

    parser.add_argument("--pred_detach",type=int,default=10)
    parser.add_argument("--equi_proj",type=str2bool,default=True)
    parser.add_argument("--equi_comb",type=str2bool,default=False)
    parser.add_argument("--pred_identity",type=str2bool,default=False)
    parser.add_argument("--neg_pos", type=int, default=1)
    parser.add_argument("--no_double", type=str2bool, default=True)
    parser.add_argument("--activation",type=str,default="relu")
    parser.add_argument("--normalizer",type=str2bool,default=False)





    #Neural network
    parser.add_argument("--tanh",type=str2bool,default=False)
    parser.add_argument("--extended",type=str2bool,default=False)
    parser.add_argument("--dropout",type=int,default=1)
    parser.add_argument("--batchnorm",type=str2bool,default=False)
    parser.add_argument("--resnet_norm",type=str,default="batchnorm")
    parser.add_argument("--proj_resnet_norm",type=str,default="batchnorm")

    parser.add_argument("--drop_val",type=float,default=0.5)
    parser.add_argument("--eval_rewards",type=str2bool,default=False)
    parser.add_argument("--average",type=str2bool,default=True)
    parser.add_argument("--max",type=str2bool,default=False)
    parser.add_argument("--channels",type=int,default=64)


    ###Environments
    parser.add_argument("--binocular",type=str2bool, default=False)
    parser.add_argument("--aperture",type=float, default=5)
    parser.add_argument("--closeness",type=float, default=0.8)
    parser.add_argument("--contrast",type=int, default=20)
    parser.add_argument("--num_actions",type=int, default=3)
    parser.add_argument("--obj_mode",type=int, default=0)
    parser.add_argument("--background",type=int, default=5)
    parser.add_argument("--noise",type=float, default=0)#3
    parser.add_argument("--rotate",type=int, default=2)
    parser.add_argument("--rotate_noise",type=float, default=0)
    parser.add_argument("--rotate_uni",type=str2bool, default=False)
    parser.add_argument("--noise_strat",type=float, default=10)
    parser.add_argument("--aug_meth",type=str, default="online")
    parser.add_argument("--quality",type=str2bool, default=False)
    parser.add_argument("--switch",type=str2bool, default=True)
    parser.add_argument("--foveation",type=int, default=0)
    parser.add_argument("--random_orient",type=str2bool, default=True)
    parser.add_argument("--cluster_mode",type=str2bool, default=False)
    parser.add_argument("--batch_mode",type=str2bool, default=False)
    parser.add_argument("--teleport",type=str2bool, default=False)
    parser.add_argument("--room_categories",type=str2bool, default=False)
    parser.add_argument("--incline",type=str2bool, default=False)


    ### Disentanglement
    parser.add_argument("--coef_base",type=float,default=1)
    parser.add_argument("--coef_independent",type=float,default=1)

    ### Augmentations
    parser.add_argument("--resize_crop",type=float,default=0)#0.08
    parser.add_argument("--flip",type=str2bool,default=False)
    parser.add_argument("--jitter",type=str2bool,default=False)
    parser.add_argument("--grayscale",type=str2bool,default=False)
    parser.add_argument("--blur",type=str2bool,default=False)
    parser.add_argument("--depth",type=float,default=0.)#0.45
    parser.add_argument("--depth_changes",type=float,default=0.075)#0.45

    parser.add_argument("--min_depth",type=float,default=0.65)
    parser.add_argument("--elevation",type=float,default=0)
    parser.add_argument("--focus",type=float,default=0)#1.4
    parser.add_argument("--focus_depth",type=str2bool,default=True)
    parser.add_argument("--min_angle_speed",type=int, default=0)
    parser.add_argument("--max_angle_speed",type=int, default=360)
    parser.add_argument("--pitch", type=float, default=0)
    parser.add_argument("--roll", type=float, default=0)
    parser.add_argument("--init_pitrol", type=str2bool, default=True)

    parser.add_argument("--def_turn", type=str2bool, default=False)
    parser.add_argument("--plabels", type=float, default=1)
    parser.add_argument("--action_predict", type=str2table, default="")

    args = parser.parse_args()


    if args.load_params is not None:
        with open(args.load_params,'r') as f:
            parsed_json=json.load(f)
            for namespace in parsed_json:
                for key in parsed_json[namespace]:
                    setattr(args, key, parsed_json[namespace][key])

    # import tools.consts as co
    # for const,val in vars(co).items():
    #     if const.startswith('_'):
    #         continue
    #     setattr(args,const,val)

    return args