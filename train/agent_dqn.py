import argparse
import time
import os
import logging as log
import math

import numpy as np

import magent

class AgentDqn:
    def __init__(self, env, handle, name,
                 batch_size = 512, target_update = 1200,
                 train_freq = 5, eval_obs = None,
                 load_from=None, savedir='save_model'):
        self.model = None
        self.savedir = savedir

        from magent.builtin.tf_model import DeepQNetwork
        self.model = DeepQNetwork(env, handle, name,
                                         batch_size=batch_size,
                                         learning_rate=3e-4,
                                         memory_size=2 ** 17, target_update=target_update,
                                         train_freq=train_freq, eval_obs=eval_obs)

        if load_from is not None:
            self.load(load_from)
        else:
            start_from = 0
    def infer_action(self, raw_obs, ids, policy="e_greedy", eps=0):
        return self.model.infer_action(raw_obs, ids, policy, eps)

    def train(self, sample_buffer, print_every=1000):
        return self.model.train(sample_buffer, print_every)

    def load(self, num_round, savedir = None):
        if savedir == None:
            savedir = self.savedir
        self.model.load(savedir, num_round)
        pass
    def save(self, num_round, savedir = None):
        if savedir == None:
            savedir = self.savedir
        self.model.save(savedir, num_round)
        pass
