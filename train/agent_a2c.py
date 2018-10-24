import argparse
import time
import os
import logging as log
import math

import numpy as np

import magent

class AgentA2c:
    def __init__(self, env, handle, name, map_size,
                 batch_size = 512, target_update = 1200, unroll_step = 8,
                 train_freq = 5, eval_obs = None,
                 load_from=None, save_dir='save_model'):
        self.model = None
        from magent.builtin.mx_model import AdvantageActorCritic
        step_batch_size = int(10 * map_size * map_size*0.01)
        self.model = AdvantageActorCritic(env, handle, name,
                                           batch_size=step_batch_size,
                                           learning_rate=2e-4)
    def load(self, savedir, num_round):

        pass
    def save(self, savedir, num_round):

        pass
