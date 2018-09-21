""" some utilities """

import math
import collections
import platform

import numpy as np
import logging
import collections
import os

import magent

class EpisodesBufferEntry(magent.utility.EpisodesBufferEntry):
    """Entry for episode buffer"""
    def __init__(self):
        super().__init__()
        self.all_views = []
        self.all_features = []
        self.all_nums = []

    def append(self, view, feature, action, reward, alive, all_view, all_feature, all_num):
        super().append(view, feature, action, reward, alive, None)
        self.all_views.append(all_view.copy())
        self.all_features.append(all_feature.copy())
        self.all_nums.append(all_num.copy())


class EpisodesBuffer(magent.utility.EpisodesBuffer):
    """Replay buffer to store a whole episode for all agents
       one entry for one agent
    """
    def record_step(self, ids, obs, acts, rewards, alives, all_obs, all_num):
        """record transitions (s, a, r, terminal) in a step"""
        buffer = self.buffer
        index = np.random.permutation(len(ids))
        if self.is_full:  # extract loop invariant in else part
            for i in range(len(ids)):
                entry = buffer.get(ids[i])
                if entry is None:
                    continue
                entry.append(obs[0][i], obs[1][i], acts[i], rewards[i], alives[i],
                             all_obs[0][i], all_obs[1][i], all_num[i])
        else:
            for i in range(len(ids)):
                i = index[i]
                entry = buffer.get(ids[i])
                if entry is None:
                    if self.is_full:
                        continue
                    else:
                        entry = EpisodesBufferEntry()
                        buffer[ids[i]] = entry
                        if len(buffer) >= self.capacity:
                            self.is_full = True

                entry.append(obs[0][i], obs[1][i], acts[i], rewards[i], alives[i],
                             all_obs[0][i], all_obs[1][i], all_num[i])

