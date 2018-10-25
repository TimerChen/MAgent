import argparse
import time
import os
import logging as log
import math

import numpy as np

import magent

class TrainerBattle:
    def __init__(self, args,
                 total_rounds = 2000,
                 rule="battle",
                 savedir='save_model', render_dir="build/render"):

        self.args = args
        self.env = None
        self.handles = None
        self.savedir = savedir
        self.start_from = None
        self.agents = []
        self.total_rounds = total_rounds

        # set logger
        log.basicConfig(level=log.INFO, filename=args.name + '.log')
        console = log.StreamHandler()
        console.setLevel(log.INFO)
        log.getLogger('').addHandler(console)

        # init the game
        self.env = env = magent.GridWorld(rule, map_size=args.map_size)
        env.set_render_dir(render_dir)

        # two groups of agents
        self.handles = handles = env.get_handles()
        self._init_a_round(env, args.map_size, handles)

        # sample eval observation set
        self.eval_obs = None
        if args.eval:
            print("sample eval set...")
            env.reset()
            self._generate_map(env, args.map_size, handles)
            self.eval_obs = magent.utility.sample_observation(env, handles, 2048, 500)[0]

    def init_agnets(self, agents, load_from=None):
        # init agents
        self.agents = agents

        if self.args.load_from is not None:
            start_from = self.args.load_from
        else:
            start_from = 0

        self.start_from = start_from

    def play(self, start_from=None, n_round=None):
        """ play and train """
        # init args
        args = self.args
        env = self.env
        handles = self.handles
        agnets = self.agents
        if start_from is None:
            start_from = self.start_from
        if n_round is None:
            n_round = args.n_round

        # print debug info
        print(args)
        print("view_space", env.get_view_space(handles[0]))
        print("feature_space", env.get_feature_space(handles[0]))

        # play
        start = time.time()
        for k in range(start_from, start_from + n_round):
            tic = time.time()
            eps = magent.utility.piecewise_decay(k,
                                                 [0, self.total_rounds*700//2000, self.total_rounds*1400//2000],
                                                 [1, 0.2, 0.05]) if not args.greedy else 0
            loss, num, reward, value = self._play_a_round(env, args.map_size, handles, self.agents,
                                                         train=args.train, print_every=50,
                                                         render=args.render or (k+1) % args.render_every == 0,
                                                         eps=eps)  # for e-greedy

            log.info("round %d\t loss: %s\t num: %s\t reward: %s\t value: %s" % (k, loss, num, reward, value))
            print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))

            # save models
            if (k + 1) % args.save_every == 0 and args.train:
                print("save model... ")
                for agent in agnets:
                    agent.save(k)

    def _generate_map(self, env, map_size, handles):
        """ generate a map, which consists of two squares of agents"""
        if True:
            env.add_agents(handles[1], method="custom", pos=[[1,2,0], [1,3,0]])
            env.add_agents(handles[0], method="custom", pos=[[1,1,0]])
            return

        if True:
            env.add_walls(method="random", n=map_size*map_size*0.04)
            env.add_agents(handles[0], method="random", n=map_size*map_size*0.05)
            env.add_agents(handles[1], method="random", n=map_size*map_size*0.01)
            return

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
        env.add_agents(handles[0], method="custom", pos=pos)

        # right
        n = init_num
        side = int(math.sqrt(n)) * 2
        pos = []
        for x in range(width//2 + gap, width//2 + gap + side, 2):
            for y in range((height - side)//2, (height - side)//2 + side, 2):
                pos.append([x, y, 0])
        env.add_agents(handles[1], method="custom", pos=pos)

    def _init_a_round(self, env, map_size, handles):
        env.reset()
        self._generate_map(env, map_size, handles)

    def _play_a_round(self, env, map_size, handles, agents, print_every,
                      max_round = 300,
                      train=True, render=False, eps=None):
        global mean_num
        self._init_a_round(env, map_size, handles)

        step_ct = 0
        done = False

        n = len(handles)
        obs  = [[] for _ in range(n)]
        obs_all = [[] for _ in range(n)]
        ids  = [[] for _ in range(n)]
        acts = [[] for _ in range(n)]
        nums = [env.get_num(handle) for handle in handles]
        nums_all = [[] for _ in range(n)]
        sample_buffer = magent.meanh_utility.EpisodesBuffer(capacity=1500)
        total_reward = [0 for _ in range(n)]

        print("===== sample =====")
        print("eps %.2f number %s" % (eps, nums))
        start_time = time.time()

        if render:
            env.render()
        while not done:
            # stat info
            nums = [env.get_num(handle) for handle in handles]

            # take actions for every model
            for i in range(n):
                obs[i] = env.get_observation(handles[i])
                ids[i] = env.get_agent_id(handles[i])
                # raw_obs = obs[i]
                # acts[i] = models[i].infer_action(obs[i], obs_all[i], nums_all[i], ids[i], 'e_greedy', eps=eps)
                acts[i] = agents[i].infer_action(obs[i], ids[i], 'e_greedy', eps=eps)
                env.set_action(handles[i], acts[i])

            # simulate one step
            done = env.step()

            # sample
            step_reward = []
            for i in range(n):
                rewards = env.get_reward(handles[i])
                if train:
                    alives = env.get_alive(handles[i])
                    sample_buffer.record_step(ids[i], obs[i], acts[i], rewards, alives, obs_all[i], nums_all[i])
                s = sum(rewards)
                step_reward.append(s)
                total_reward[i] += s


            nums = [env.get_num(handle) for handle in handles]

            # render
            if render:
                env.render()

            # clear dead agents
            env.clear_dead()

            if step_ct % print_every == 0:
                print("step %3d,  nums: %s reward: %s,  total_reward: %s " %
                      (step_ct, nums, np.around(step_reward, 2), np.around(total_reward, 2)))
            step_ct += 1
            if step_ct > max_round:
                break

        sample_time = time.time() - start_time
        print("steps: %d,  total time: %.2f,  step average %.2f" % (step_ct, sample_time, sample_time / step_ct))

        # train
        total_loss, value = 0, 0
        if train:
            print("===== train =====")
            start_time = time.time()
            total_loss, value = agents[0].train(sample_buffer, 1000)
            train_time = time.time() - start_time
            print("train_time %.2f" % train_time)

        def round_list(l): return [round(x, 2) for x in l]
        return total_loss, nums, round_list(total_reward), value