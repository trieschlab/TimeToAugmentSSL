import datetime
import os
from time import sleep

import gym
import matplotlib
import torch
import torchvision
from tdw.tdw_utils import TDWUtils
from torch import nn
from torch.utils.data import DataLoader

from envs.move_play import MovePlay
from envs.room_toys import RoomToys
from envs.full_play import FullPlay
from envs.six_objects import get_label_names, ToysObjects, ContinuousToyObjects
from models.dqn import Dqn
from models.rep_models import LearnRep
from models.replay import Replay
from models.resnets import get_network
from tools import utils
from tools.arguments import parse
from tools.evaluation import save_image, lls, knn_evaluation, get_representations
from tools.finetuning import finetune
from tools.logger import EpochLogger
from tools.logging import DatasetHandler, LogTrainer
from tools.utils import size_space, build_default_act_space, get_augmentations, encode_actions, dim_space, \
    get_standard_dataset, TdwImageDataset, TestImageDataset
from tools.wrappers import TimeLimitSpe, WrapPyTorch, ContinuousToDiscrete, RestrictContinuous, OrdRestrictContinuous
from models.networks import MLP, ConvNet128, ConvMLP, Actor_SAC
from models.sac import SAC
from models.simple_agents import *

args = parse()
num_threads = 1 if args.device == "cpu" else 4
str_threads=str(num_threads)
os.environ["OMP_NUM_THREADS"] = str_threads
os.environ["OPENBLAS_NUM_THREADS"] = str_threads
os.environ["MKL_NUM_THREADS"] = str_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = str_threads
os.environ["NUMEXPR_NUM_THREADS"] = str_threads
os.environ["DISPLAY"] = args.display #args.display + str(gpu_id)


def main():
    ###Prepare logs and cuda
    save_dir= args.save+datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")+'_'+str(args.seed)+"-"+args.exp_name
    # if not os.path.exists(save_dir+"/clusters/"):
    # os.mkdir(save_dir)
    os.makedirs(save_dir+"/clusters")
    os.makedirs(save_dir+"/clusters_back")
    os.makedirs(save_dir+"/clusters_pos")

    gym.logger.set_level(40)
    torch.set_printoptions(edgeitems=4)

    utils.prepare_device(args)
    if args.local:
        TDWUtils.set_default_libraries(model_library=os.environ["LOCAL_BUNDLES"] + "/local_asset_bundles/models.json",
                                       scene_library=os.environ["LOCAL_BUNDLES"] + "/local_asset_bundles/scenes.json",
                                       material_library=os.environ["LOCAL_BUNDLES"] + "/local_asset_bundles/materials.json",
                                       hdri_skybox_library=os.environ["LOCAL_BUNDLES"] + "/local_asset_bundles/hdri_skyboxes.json")
    print(os.environ["LOCAL_BUNDLES"] + "/local_asset_bundles/models.json")

    dataset_handler = DatasetHandler(args, save_dir)
    env = get_env_name(args)
    env = WrapPyTorch(TimeLimitSpe(env, max_episode_steps=args.reset, reset=args.reset_correction), squeeze=False)
    if args.def_turn: env = OrdRestrictContinuous(env, args, n=args.nord)
    else: env = RestrictContinuous(env, args)
    # if args.agent == "dqn" and args.env_name == "full_play": env = ContinuousToDiscrete(env, args)
    hd = HandDefined(args)
    # if args.agent == "sac": env = RestrictContinuous(env, args)

    ###Init models
    buffer = Replay(args, env.observation_space, env.action_space)
    agent = get_agent(args,env.action_space, buffer)
    embed_network = get_network(args)
    print(embed_network)
    rep_function = LearnRep(env.action_space, args, embed_network)

    ###Start training loop
    full_obs = env.reset()
    prev_obs = full_obs["observation"]
    prev_prev_obs=prev_obs
    prev_actions = torch.zeros(dim_space(env.action_space),device=args.device)
    prev_prev_prev_obs=prev_obs
    i=0
    cpt_train = 0
    cpt_tests = 0
    time_for_step = 0
    train_Logger=LogTrainer(args, agent, rep_function, buffer, save_dir)
    rep_function.load()
    buffer.load()

    minimum_time = args.minimum_time
    sample = None
    # test_func = test if "independent" not in args.regularizers else test_double
    for e in range(args.start_epochs,args.num_epochs):
        # if e != 0:
        dataset_handler.tests(rep_function, cpt_tests)
        cpt_tests +=1
        for i in range(args.num_updates):
            if buffer.total_size < minimum_time + 1 or (args.separate and (cpt_train < 0.5* args.num_epochs * args.num_updates)):
                a = agent.act(None, rand=True, full_obs=full_obs)
            else:
                a = agent.act(get_state(rep_function, prev_obs, prev_prev_obs, prev_prev_prev_obs, prev_actions, sample, env.action_space, full_obs), steps=cpt_train, full_obs=full_obs)
            a = hd.act(a)
            t = time.time()
            full_obs, rew, done, info = env.step(a.cpu()) #time 0.05
            time_for_step += time.time()-t
            obs = full_obs["observation"]
            if args.reward_type == 9:
                with torch.no_grad():rew = get_pos_reward(rep_function, prev_obs, obs)
            if info["prev_obs"] is not None:
                prev_obs = torch.tensor(info["prev_obs"])
            buffer.insert(prev_obs, obs, rew, done, a.cpu(), info=info, p_a=agent.p_a, prev_obs=prev_prev_obs, prev_prev_obs=prev_prev_prev_obs, prev_action=prev_actions) #time e-5
            if cpt_train < 25:
                save_image(save_dir+"/example_images", cpt_train, prev_obs,args=args)
                if info["true_obs"] is not None:
                    save_image(save_dir + "/example_images", str(cpt_train)+"_2", info["true_obs"].unsqueeze(0),args=args)

            cpt_train +=1
            prev_prev_prev_obs= obs if info["true_obs"] is not None else prev_prev_obs
            prev_prev_obs= obs if info["true_obs"] is not None else prev_obs
            prev_obs = obs
            prev_actions = a

            if buffer.total_size > 5*args.batch_size:
                # rew_logger.update_stats(sample)
                for _ in range(args.train_per_step):
                    sample = train(agent, rep_function, buffer, cpt_train)  # time 0.16
                if sample is not None:
                    try:
                        train_Logger.update(info, a, sample)
                        time_for_step = train_Logger.log(time_for_step,cpt_train, info, sample)
                    except:
                        print("error rew log")

        if args.multisave and e%5 == 0 and e > 0:
            rep_function.save(save_dir)
            agent.save(save_dir)
            buffer.save(save_dir)

    dataset_handler.tests(rep_function,cpt_tests)

    rep_function.save(save_dir)
    agent.save(save_dir)
    buffer.save(save_dir)
    print("before closing")
    env.close()

    finetune(args, dataset_handler.datasets[0]["dataset"], dataset_handler.datasets[1]["dataset"], rep_function.net, save_dir)



def train(agent, rep_function, buffer, step_train):
    to_update = (step_train % args.nlearn == 0)
    sample = None
    if to_update and args.nlearn != -1:
        sample = buffer.sample()
        irewards, loss = rep_function.learn(sample)
        if not args.separate or (step_train >= 0.5* args.num_epochs * args.num_updates):
            with torch.no_grad():
                next_q_value = agent.get_value(sample)
            returns = (next_q_value* args.gamma + irewards).detach()
            agent.evaluate(sample, returns)
        if not args.separate or (step_train < 0.5*args.num_epochs * args.num_updates) or args.reward_type == 9:
            rep_function.update(loss, sample)
    return sample

def get_pos_reward(rep_function, obs1, obs2):
    if rep_function.args.eval_rewards:
        rep_function.net.eval()
    embs, _ = rep_function.embed(torch.cat((obs1, obs2), dim=0))
    if rep_function.args.eval_rewards:
        rep_function.net.train()
    return rep_function.sim_function_simple(embs[0:1], embs[1:2])/rep_function.args.temperature

def get_state(rep_function, prev_obs, prev_prev_obs, prev_prev_prev_obs, prev_actions, sample, action_space,full_obs):
    if args.eval_rewards or args.architecture == "resnet18":
        rep_function.net.eval()
        if rep_function.net_target is not None:
            rep_function.net_target.eval()

    if args.drl_inputs == "rewards":
        embs, _ = rep_function.embed(torch.cat((prev_obs, prev_prev_obs), dim=0))
        reward = rep_function.get_loss(embs[0:1], embs[1:2], sample["true_embed"][1:])
        if args.normalizer:
            reward = rep_function.normalize.apply(reward)
        s= rep_function.drl_embed(prev_obs, reward=reward).to(args.device)
    elif args.drl_inputs == "views":
        s= torch.tensor([float(full_obs["angle"])/180. - 1], device=args.device).view(1,1)
    elif args.drl_inputs == "views2":
        s= torch.tensor([math.cos(float(full_obs["angle"])*math.pi/180),math.sin(float(full_obs["angle"])*math.pi/180)], device=args.device).view(1,2)
    elif args.drl_inputs == "prev_rewards" or args.drl_inputs == "diff_rewards":
        embs, _ = rep_function.embed(torch.cat((prev_obs, prev_prev_obs, prev_prev_prev_obs), dim=0))
        prev_loss = rep_function.get_loss(embs[0:1], embs[1:2], sample["true_embed"][1:])
        prev_prev_loss = rep_function.get_loss(embs[1:2], embs[2:3], sample["true_embed"][1:])
        # print("act:", prev_loss.item(), prev_prev_loss.item(), encode_actions(prev_actions, action_space))
        if args.normalizer:
            prev_loss = rep_function.normalizer.apply(prev_loss)
            prev_prev_loss = rep_function.normalizer.apply(prev_prev_loss)
        if args.drl_inputs == "prev_rewards":
            s =torch.cat((prev_loss, prev_prev_loss, encode_actions(prev_actions, action_space).to(args.device).view(1, -1)),dim=1)
        elif args.drl_inputs == "diff_rewards":
            s = torch.cat((prev_loss - prev_prev_loss, encode_actions(prev_actions, action_space).to(args.device).view(1, -1)),dim=1)

    else:
        s = rep_function.drl_embed(prev_obs)
    if args.eval_rewards or args.architecture == "resnet18":
        rep_function.net.train()
        if rep_function.net_target is not None:
            rep_function.net_target.train()

    return s

def get_agent(args, action_space, buffer):
    try:
        name, num_fix = args.agent.split("_", 1)
    except:
        name, num_fix = args.agent, 0

    if name == "ordturn":
        name, num_fix, p_turn = args.agent.split("_", 2)
        agent = NordPturn(int(num_fix), float(p_turn), action_space)
        return agent
    rm = 0
    if num_fix != 0:rm = 1
    dim_action = size_space(action_space) - rm
    if name == "dqn":
        action_space = gym.spaces.Discrete(2)
        if args.drl_inputs == "inputs":
            critic_network = ConvMLP(args,num_output=size_space(action_space), hidden_size=args.neurons,activation=nn.ELU, num_layers=args.layers, last_linear=True).to(args.device)
        elif args.drl_inputs == "prev_rewards":
            critic_network = MLP(num_inputs=1 + dim_action + 1, num_output=size_space(action_space), hidden_size=args.neurons,activation=nn.ELU, num_layers=args.layers, last_linear=True).to(args.device)
        elif args.drl_inputs in ["diff_rewards","rewards","views"]:
            critic_network = MLP(num_inputs=1 if args.drl_inputs != "diff_rewards" else 1+ dim_action , num_output=size_space(action_space), hidden_size=args.neurons,activation=nn.ELU, num_layers=args.layers, last_linear=True).to(args.device)
        elif args.drl_inputs == "views2":
            critic_network = MLP(num_inputs=2 , num_output=size_space(action_space), hidden_size=args.neurons,activation=nn.ELU, num_layers=args.layers, last_linear=True).to(args.device)
        else:
            if args.drl_inputs != "projmix" and args.architecture == "resnet18":
                input_size = 512
            else:
                input_size = args.num_latents
            critic_network = MLP(num_inputs=input_size, num_output=size_space(action_space), hidden_size=args.neurons,activation=nn.ELU, num_layers=args.layers, last_linear=True).to(args.device)
        agent = Dqn(args, critic_network, action_space)
    elif name == "sac":
        critic2 = None
        if args.drl_inputs == "inputs":
            critic_network = ConvMLP(extra=dim_action, num_output=1, hidden_size=args.neurons,activation=nn.ELU, num_layers=args.layers, last_linear=True).to(args.device)
            actor_network = ConvMLP(num_output=dim_action, hidden_size=args.neurons,activation=nn.ELU, num_layers=args.layers, last_linear=True).to(args.device)
            if args.double_critic: critic2 = ConvMLP(extra=dim_action, num_output=1, hidden_size=args.neurons,activation=nn.ELU, num_layers=args.layers, last_linear=True).to(args.device)
        elif args.drl_inputs in ["diff_rewards","rewards","views","prev_rewards","views2"]:
            more_dim = 0
            if args.drl_inputs == "prev_rewards":
                more_dim = dim_action + 1
            if args.drl_inputs == "views2":
                more_dim +=1
            critic_network = MLP(num_inputs=1 + dim_action + more_dim, num_output=1,hidden_size=args.neurons, activation=nn.ELU, num_layers=args.layers,last_linear=True).to(args.device)
            actor_network = MLP(num_inputs=1+ more_dim, num_output=dim_action , hidden_size=args.neurons, activation=nn.ELU, num_layers=args.layers, last_linear=True).to(args.device)
            if args.double_critic:critic2 = MLP(num_inputs=1 + dim_action + more_dim, num_output = 1, hidden_size = args.neurons, activation = nn.ELU, num_layers = args.layers, last_linear = True).to(args.device)
        else:
            if args.drl_inputs != "projmix":
                input_size = 512
            else:
                input_size = args.num_latents
            critic_network = MLP(num_inputs=input_size+dim_action, num_output=1, hidden_size=args.neurons,activation=nn.ELU, num_layers=args.layers, last_linear=True).to(args.device)
            actor_network = MLP(num_inputs=input_size, num_output=dim_action, hidden_size=args.neurons,activation=nn.ELU, num_layers=args.layers, last_linear=True).to(args.device)
            if args.double_critic:critic2 = MLP(num_inputs=args.num_latents + dim_action, num_output = 1, hidden_size = args.neurons, activation = nn.ELU, num_layers = args.layers, last_linear = True).to(args.device)
        actor = Actor_SAC(actor_network, action_space, args).to(args.device)
        agent = SAC(args, action_space, critic_network, actor, critic2, num_fix)
    elif name == "fix":
        agent = Nfix(int(num_fix), action_space)
    elif name == "rnd":
        agent = Rnd(action_space)
    elif name == "currord":
        agent = CurrOrd(action_space)
    elif name == "freqobj":
        agent = FreqAgentObj(buffer, action_space)
    elif name == "freqcat":
        agent = FreqAgentCat(buffer, action_space)
    elif name == "pord":
        agent = Pord(args.def_turn, int(num_fix), action_space)
    elif name == "ord":
        agent = Nord(args.def_turn, int(num_fix), action_space)
    elif name == "bad":
        agent = Left(action_space)
    elif name == "play":
        agent = Play(action_space)
    return agent

def get_env_name(args, **kwargs):
    if args.env_name == "tdw_toys":
        env = ToysObjects(args, **kwargs)
    elif args.env_name == "tdw_toys_cont":
        env = ContinuousToyObjects(args, **kwargs)
    elif args.env_name == "move_play":
        default_actions = build_default_act_space(args)
        env = MovePlay(args, default_values=default_actions, **kwargs)
    elif args.env_name == "full_play":
        default_actions = build_default_act_space(args)
        env = FullPlay(args, default_values=default_actions, **kwargs)
    elif args.env_name == "tdw_room_toys":
        env = RoomToys(args)
    return env



if __name__ == "__main__":
    main()
    #KeyboardControls().run()