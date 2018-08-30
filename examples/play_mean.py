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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def generate_map(env, map_size, handles, leftID=0, rightID=1):
    """ generate a map, which consists of two squares of agents"""
    width = height = map_size
    init_num = map_size * map_size * 0.04
    gap = 3

    # left
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 - gap - side, width//2 - gap - side + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 + gap, width//2 + gap + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[rightID], method="custom", pos=pos)


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
    nums = [env.get_num(handle) for handle in handles]
    sample_buffer = magent.utility.EpisodesBuffer(capacity=1500)
    total_reward = [0 for _ in range(n)]

    start_time = time.time()
    while not done:
        # take actions for every model
        for i in range(n):
            obs[i] = env.get_observation(handles[i])
            if types[i] == 'mean_action':
                mean_obs[i] = env.get_mean_action(handles[i])
                obs[i] = (obs[i][0], np.concatenate([obs[i][1],mean_obs[i]], axis=1))
            elif types[i] == 'mean_all':
                mean_obs[i] = env.get_mean_observation(handles[i])
                obs[i] = (np.concatenate([obs[i][0],mean_obs[i][0]], axis=3),
                          np.concatenate([obs[i][1],mean_obs[i][1]], axis=1))
            ids[i] = env.get_agent_id(handles[i])
            models[i].infer_action(obs[i], ids[i], 'e_greedy', eps=eps, block=False)
        for i in range(n) :
            acts[i] = models[i].fetch_action()
            env.set_action(handles[i], acts[i])

        done = env.step()
        nums = [env.get_num(handle) for handle in handles]
        env.clear_dead()

        step_ct += 1
        if step_ct > 550:
            break
    return nums

def extract_model_names(savedir, name, model_class, type=None, begin=0, pick_every=4):
    if model_class is DeepQNetwork:
        prefix = 'tfdqn'
    if type == None:
        type = name

    pattern = re.compile(prefix + '_(\d*).meta')

    ret = []
    for path in os.listdir(os.path.join(savedir, name)):
        match = pattern.match(path)
        if match and int(match.group(1)) > begin:
            ret.append((savedir, name, int(match.group(1)), model_class, type))

    ret.sort(key=lambda x: x[2])
    ret = [ret[i] for i in range(0, len(ret), pick_every)]

    return ret

def play_wrapper(model_names, n_rounds=6):
    time_stamp = time.time()

    models = []
    for i, item in enumerate(model_names):

        view_space, feature_space = None, None
        if item[-1] == 'mean_action':
            feature_space = (env.get_feature_space(handles[0])[0] + env.get_mean_action_space(handles[0])[0],)
        elif item[-1] == 'mean_all':
            view_space = env.get_view_space(handles[0])
            view_space = view_space[0:2] + (view_space[2] + env.get_mean_view_space(handles[0])[2], )
            feature_space = env.get_feature_space(handles[0])
            feature_space = (feature_space[0] + env.get_mean_feature_space(handles[0])[0],)
        models.append(magent.ProcessingModel(env, handles[i], item[1], 0, RLModel=item[-2],
                                             custom_view_space = view_space, custom_feature_space = feature_space))

    for i, item in enumerate(model_names):
        models[i].load(item[0], item[2])

    types = [item[-1] for item in model_names]
    leftID, rightID = 0, 1
    result = 0
    total_num = np.zeros(2)
    for _ in range(n_rounds):
        round_num = play_a_round(env, handles, models, types, args.map_size, leftID, rightID)
        total_num += round_num
        leftID, rightID = rightID, leftID
        result += 1 if round_num[0] > round_num[1] else 0
    result = 1.0 * result

    for model in models:
        model.quit()

    return result / n_rounds, total_num / n_rounds, time.time() - time_stamp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--render_every", type=int, default=10)
    parser.add_argument("--n_round", type=int, default=2000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--map_size", type=int, default=63)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--name", type=str, default="battle")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument('--alg', default='dqn', choices=['dqn', 'drqn'])
    args = parser.parse_args()


    # init the game
    env = magent.GridWorld("battle", map_size=args.map_size)
    env.set_render_dir("build/render")

    handles = env.get_handles()
    init_a_round(env, args.map_size, handles)

    model_name = []
    model_name = model_name + extract_model_names('save_model', 'single_base_mini', DeepQNetwork, type='simple',begin=1399, pick_every=1)
    print('number of models', len(model_name))
    model_name = model_name + extract_model_names('save_model', 'mean_action', DeepQNetwork, begin=1399, pick_every=1)
    print('number of models', len(model_name))
    model_name = model_name + extract_model_names('save_model', 'mean_all', DeepQNetwork, begin=1399, pick_every=1)
    print('number of models', len(model_name))


    # print debug info
    print(args)

    detail_file = open("detail.log", "w")
    winrate_file = open("win_rate.log", "w")

    n = len(model_name)
    rate = [[0.0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(i,n):
            rate[i][j], nums, elapsed = play_wrapper([model_name[i], model_name[j]], 6)
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