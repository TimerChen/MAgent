"""
Train a single model by self-play
"""


import argparse
import time
import os
import logging as log
import math

import numpy as np

import magent

import agent_dqn
import agent_rulebase
import trainer_battle


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
    parser.add_argument("--name", type=str, default="shepherd_dqn")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument('--alg', default='dqn', choices=['dqn', 'drqn','a2c'])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    trainer = trainer_battle.TrainerBattle(args)
    agents = [
        # agent_rulebase.AgentRulebase(trainer.env, trainer.handles[0], type="RushGatherer"),
        # agent_dqn.AgentDqn(trainer.env, trainer.handles[0], "battle_dqn"),
        agent_rulebase.AgentRulebase(trainer.env, trainer.handles[0], trainer.handles[1], type="NoMove", blind=True),
        agent_rulebase.AgentRulebase(trainer.env, trainer.handles[1], trainer.handles[0], type="Rush", blind=False),
        #agent_dqn.AgentDqn(trainer.env, trainer.handles[1], args.name)
              ]
    trainer.init_agnets(agents)

    trainer.play()



