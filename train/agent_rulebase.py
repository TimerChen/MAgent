import argparse
import time
import os
import logging as log
import math

import numpy as np

import magent

class AgentRulebase:
    def __init__(self, env, handle, target_handle=None, type="Random", blind=False):
        self.model = None

        #from magent.builtin.rule_model import RushPredator
        from magent.builtin.rule_model import RushGatherer
        from magent.builtin.rule_model import Run
        #from magent.builtin.rule_model import RunawayPrey
        from magent.builtin.rule_model import RandomActor
        if type == "Rush":
            self.model = RushGatherer(env, handle, blind=blind)
        elif type == "Run":
            self.model = Run(env, handle, blind=blind)
        else:
            self.model = RandomActor(env, handle)

    def infer_action(self, raw_obs, ids, policy="e_greedy", eps=0):
        return self.model.infer_action(raw_obs, ids, policy, eps)

    def train(self, sample_buffer, print_every=1000):
        return (0.0, 0.0)

    def load(self, num_round, savedir = None):
        pass

    def save(self, num_round, savedir = None):
        pass
