"""
Train a single model by self-play
"""


import argparse
import time
import os
import logging as log
import math
import re

import numpy as np

import magent
from magent.builtin.tf_model import DeepQNetwork
from magent.builtin.tf_model import DeepQNetwork_MC
from magent.builtin.tf_model import DeepQNetwork_meanh
from magent.builtin.tf_model import AdvantageActorCritic

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mean_num = [0,0]

def generate_map(env, map_size, handles, leftID=0, rightID=1):
    """ generate a map, which consists of two squares of agents"""
    global mean_num
    width = height = map_size
    init_num = map_size * map_size * 0.04
    gap = 1

    # left
    n = init_num
    side = int(math.sqrt(n)) * 2
    wside = 6
    hside = 10
    pos = []
    for x in range(width//2 - gap - wside, width//2 - gap - wside + wside, 2):
        for y in range((height - hside)//2+1, (height - hside)//2 + hside, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[0], method="custom", pos=pos)

    #right
    pos = [[width-i[0]-1, i[1], 0] for i in pos]
    env.add_agents(handles[1], method="custom", pos=pos)


def init_a_round(env, map_size, handles):
    env.reset()
    generate_map(env, map_size, handles)

def play_a_round(env,  handles, models, types, map_size ,print_every, train=True, render=False, eps=0.05):
    init_a_round(env, map_size, handles)

    step_ct = 0
    done = False

    n = len(handles)
    obs  = [[] for _ in range(n)]
    mean_obs = [[] for _ in range(n)]
    ids  = [[] for _ in range(n)]
    acts = [[] for _ in range(n)]
    speak_channel = [[] for _ in range(n)]
    obs_all = [[] for _ in range(n)]
    nums_all = [[] for _ in range(n)]
    nums = [env.get_num(handle) for handle in handles]
    sample_buffer = magent.utility.EpisodesBuffer(capacity=1500)
    total_reward = [0 for _ in range(n)]

    if render:
        env.render()

    start_time = time.time()
    while not done:
        nums = [env.get_num(handle) for handle in handles]

        # take actions for every model
        for i in range(n):
            obs[i] = env.get_observation(handles[i])
            if types[i] == 'mean_action' or types[i] == 'multi-chan':
                mean_obs[i] = env.get_mean_action(handles[i])
                obs[i] = (obs[i][0], np.concatenate([obs[i][1],mean_obs[i]], axis=1))
            elif types[i] == 'mean_all':
                mean_obs[i] = env.get_mean_observation(handles[i])
                obs[i] = (np.concatenate([obs[i][0],mean_obs[i][0]], axis=3),
                          np.concatenate([obs[i][1],mean_obs[i][1]], axis=1))
            elif types[i] == 'meanh':
                raw_obs = obs[i]
                padding_num = mean_num[i] - nums[i]
                assert raw_obs[0].shape[0] == nums[i]
                assert padding_num >= 0
                if padding_num > 0:
                    obs_all[i] = (np.concatenate([raw_obs[0], np.zeros((padding_num,)+raw_obs[0].shape[1:])]),
                                  np.concatenate([raw_obs[1], np.zeros((padding_num,)+raw_obs[1].shape[1:])]) )
                else:
                    obs_all[i] = raw_obs
                obs_all[i] = (np.array([obs_all[i][0] for j in range(nums[i])]),
                              np.array([obs_all[i][1] for j in range(nums[i])]))
                nums_all[i] = [[nums[i]] for j in range(nums[i])]
                nums_all[i] = np.array(nums_all[i], dtype=np.int32)

            ids[i] = env.get_agent_id(handles[i])
            if types[i] == 'meanh':
                acts[i] = models[i].infer_action_all(obs[i], obs_all[i], nums_all[i], ids[i], 'e_greedy', eps=eps)
            else:
                acts[i] = models[i].infer_action(obs[i], ids[i], 'e_greedy', eps=eps)

            env.set_action(handles[i], acts[i])
            if types[i] == 'multi-chan':
                speak_channel[i] = models[i].infer_speak_channel(obs[i], ids[i], 'e_greedy', eps=eps)
                env.set_speak_channel(handles[i], speak_channel[i])
        done = env.step()
        if render:
            env.render()
        env.clear_dead()

        step_ct += 1
        if step_ct > 250:
            break
    return nums

def extract_model_names(savedir, name, model_class, type=None, begin=0, pick_every=4, end=None):
    if model_class is DeepQNetwork:
        prefix = 'tfdqn'
    elif model_class is DeepQNetwork_MC:
        prefix = 'tfdqn_mc'
    elif model_class is AdvantageActorCritic:
        prefix = 'tfa2c'
    elif model_class is DeepQNetwork_meanh:
        prefix = 'tfdqn_mean_h'
    if type == None:
        type = name

    pattern = re.compile(prefix + '_(\d*).meta')

    ret = []
    for path in os.listdir(os.path.join(savedir, name)):
        match = pattern.match(path)
        if match and int(match.group(1)) >= begin:
            if end is not None and int(match.group(1)) > end:
                continue
            ret.append((savedir, name, int(match.group(1)), model_class, type))

    ret.sort(key=lambda x: x[2])
    ret = [ret[i] for i in range(0, len(ret), pick_every)]

    return ret

def play_wrapper(model_names, n_rounds=20):
    time_stamp = time.time()

    models = []
    now_handles = handles
    for i, item in enumerate(model_names):

        view_space, feature_space = None, None
        if item[-1] == 'mean_action' or item[-1] == 'multi-chan':
            feature_space = (env.get_feature_space(now_handles[i])[0] + env.get_mean_action_space(now_handles[i])[0],)
        elif item[-1] == 'mean_all':
            view_space = env.get_view_space(now_handles[i])
            view_space = view_space[0:2] + (view_space[2] + env.get_mean_view_space(now_handles[i])[2], )
            feature_space = env.get_feature_space(now_handles[i])
            feature_space = (feature_space[0] + env.get_mean_feature_space(now_handles[i])[0],)
        #print(view_space, feature_space)
        #print(env.get_view_space(now_handles[i]), env.get_feature_space(now_handles[i]))
        #print('models.append', now_handles[i], item[1], 0, item[-2])
        if item[-1] == 'meanh':
            if item[1] == 'meanh_42':
                h_size = 42
            elif item[1] == 'meanh_10':
                h_size = 10
            else:
                h_size = 21
            models.append(magent.ProcessingModel(env, now_handles[i], item[1], 21000+i, RLModel=item[-2],
                                             custom_view_space = view_space, custom_feature_space = feature_space,
                                                 mean_num=max(mean_num[0], mean_num[1]), h_size = h_size,
                                                 memory_size=2 ** 17))
        else:
            models.append(magent.ProcessingModel(env, now_handles[i], item[1], 21000+i, RLModel=item[-2],
                                                 custom_view_space = view_space, custom_feature_space = feature_space,
                                                 memory_size=2 ** 18))

    for i, item in enumerate(model_names):
        models[i].load(item[0], item[2])

    types = [item[-1] for item in model_names]
    leftID, rightID = 0, 1
    result = 0
    total_num = np.zeros(2)
    for _ in range(n_rounds):
        round_num = play_a_round(env, now_handles, models, types, args.map_size, leftID, rightID, render=args.render)
        total_num += round_num
        leftID, rightID = rightID, leftID
        result += 1 if round_num[0] > round_num[1] else 0
    result = 1.0 * result

    for model in models:
        model.quit()

    return result / n_rounds, total_num / n_rounds, time.time() - time_stamp

""" battle of two armies """
def get_config(map_size, leftType, rightType):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})
    cfg.set({"embedding_size": 10})

    multi_channel = cfg.register_agent_type(
        "multi_channel",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
         'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(1.5),
         'damage': 2, 'step_recover': 0.1,

         #for multi-channel communication
         'comm_channel': 2, 'hear_all_group': False,

         'step_reward': -0.005,  'kill_reward': 5, 'dead_penalty': -0.1, 'attack_penalty': -0.1,
         })
    small = cfg.register_agent_type(
        "small",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
         'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(1.5),
         'damage': 2, 'step_recover': 0.1,

         'step_reward': -0.005,  'kill_reward': 5, 'dead_penalty': -0.1, 'attack_penalty': -0.1,
         })

    g0 = cfg.add_group(small if leftType != 'multi-chan' else multi_channel)
    g1 = cfg.add_group(small if rightType != 'multi-chan' else multi_channel)

    a = gw.AgentSymbol(g0, index='any')
    b = gw.AgentSymbol(g1, index='any')

    # reward shaping to encourage attack
    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=0.2)
    cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=b, value=0.2)

    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--render_every", type=int, default=10)
    parser.add_argument("--n_round", type=int, default=2000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--map_size", type=int, default=13)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--name", type=str, default="battle")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument('--alg', default='dqn', choices=['dqn', 'drqn'])
    args = parser.parse_args()


    env = None

    model_name = []


    # model_name = model_name + extract_model_names('save_model', 'meanh_10', DeepQNetwork_meanh, begin=1399, pick_every=1,
    #                                                type='meanh')
    # print('number of models', len(model_name))
    # model_name = model_name + extract_model_names('save_model', 'meanh_42', DeepQNetwork_meanh, begin=1399, pick_every=1,
    #                                                type='meanh')
    # print('number of models', len(model_name))
    model_name = model_name + extract_model_names('save_model', 'mf_mini', DeepQNetwork, begin=1499, end=1499, pick_every=1,
                                                  type='mean_action')
    print('number of models', len(model_name))

    model_name = model_name + extract_model_names('save_model', 'meanh', DeepQNetwork_meanh, begin=1899, end=1899, pick_every=1)
    print('number of models', len(model_name))

    #model_name = model_name + extract_model_names('save_model', 'single_base_mini', DeepQNetwork, begin=1399, pick_every=1)
    #print('number of models', len(model_name))





    # print debug info
    print(args)

    detail_file = open("detail.log", "w")
    winrate_file = open("win_rate.log", "w")

    n = len(model_name)
    rate = [[0.0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            # init the game
            env = magent.GridWorld(get_config(args.map_size, model_name[i][-1], model_name[j][-1]),
                                   map_size=args.map_size)
            env.set_render_dir("build/render")
            handles = env.get_handles()
            init_a_round(env, args.map_size, handles[:2])
            mean_num = [env.get_num(handle) for handle in handles]
            print(mean_num)

            rate[i][j], nums, elapsed = play_wrapper([model_name[i], model_name[j]], 20)
            rate[j][i] = 1.0 - rate[i][j]
            round_res = ("model1: %s\t model2: %s\t rate: %.2f\t num: %s\t elapsed: %.2f" %
                         (model_name[i][:-2], model_name[j][:-2], rate[i][j], list(nums), elapsed))
            print('-------[',i,j,']-------')
            print(round_res)

            detail_file.write(round_res + "\n")
            detail_file.flush()
        winrate_file.write("model: %s\twin rate: %.2f\n" % (model_name[i],
                                                            1.0 * sum(np.asarray(rate[i])) / (len(model_name) - 1)))

        winrate_file.flush()
