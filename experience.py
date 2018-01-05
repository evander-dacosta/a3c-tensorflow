#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:32:43 2018

@author: evanderdcosta
"""

class Experience:
    def __init__(self, action, state, reward, terminal):
        self.action = action
        self.state = state
        self.reward = reward
        self.terminal = terminal