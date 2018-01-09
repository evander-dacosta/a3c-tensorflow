#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:32:43 2018

@author: evanderdcosta
"""

class Experience:
    def __init__(self, state, action, reward, next_state, terminal):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminal = terminal
        
        
    def get(self):
        return self.state, self.action, self.reward, self.next_state, \
                self.terminal