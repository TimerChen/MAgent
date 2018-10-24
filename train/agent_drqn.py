import argparse
import time
import os
import logging as log
import math

import numpy as np

import magent

class AgentDrqn:
    def __init__(self, env, handle, name,
                 batch_size = 512, target_update = 1200, unroll_step = 8,
                 train_freq = 5, eval_obs = None,
                 load_from=None, save_dir='save_model'):
        self.model = None
        from magent.builtin.tf_model import DeepRecurrentQNetwork
        self.model = DeepRecurrentQNetwork(env, handle, name,
                                            learning_rate=3e-4,
                                            batch_size=batch_size//unroll_step, unroll_step=unroll_step,
                                            memory_size=2 * 8 * 625, target_update=target_update,
                                            train_freq=train_freq, eval_obs=eval_obs)

    def load(self, savedir, num_round):

        pass
    def save(self, savedir, num_round):

        pass
