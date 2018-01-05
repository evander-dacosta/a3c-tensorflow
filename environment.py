#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:33:05 2018

@author: evanderdcosta
"""

import gym
import numpy as np

class Environment(object):
    def __init__(self, config):
        self.env = gym.make(config.env_name)
        
        self.random_start_steps, self.display = \
              config.random_start_steps, config.display
        
        self._state = None
        self.reward = 0
        self.terminal = True
        
    def reset(self):
        """
        Reset the environment
        """
        self._state = self.env.reset()
        self.render()
        self.terminal = False
        return self.state, 0, 0, self.terminal
    
    def random_start(self):
        """
        Starts a game and plays a few frames at random. 
        Great for starting at different random positions.
        
        Of course, there is a danger that your game just keeps
        starting at terminal positions. If this is the case,
        reduce 'config.random_start'
        """
        self.reset()
        for _ in range(np.random.randint(0, self.random_start_steps - 1)):
            self.step(self.random_step())
        self.render()
        return self.state, 0, 0, self.terminal
    
    def step(self, action):
        self._state, self.reward, self.terminal, _ = self.env.step(action)
        
    def random_step(self):
        action = self.env.action_space.sample()
        return action
        
    @property
    def state(self):
        return self.preprocess(self._state)
    
    @property
    def action_size(self):
        return self.env.action_space.n
    
    def render(self):
        if(self.display):
            self.env.render()
            
    def preprocess(self, state):
        raise NotImplementedError()
        
        

        
class SimpleEnvironment(Environment):
    def __init__(self, config):
        super(SimpleEnvironment, self).__init__(config)
        
    def preprocess(self, state):
        return state