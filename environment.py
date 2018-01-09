#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:33:05 2018

@author: evanderdcosta
"""

import gym
import numpy as np
from utils import imresize, rgb2gray

class Environment(object):
    def __init__(self, config):
        self.env = gym.make(config.env_name)
        self.env_name = config.env_name
        
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
        self.reward = 0
        self.terminal = False
        self.render()
    
    def random_start(self):
        """
        Starts a game and plays a few frames at random. 
        Great for starting at different random positions.
        
        Of course, there is a danger that your game just keeps
        starting at terminal positions. If this is the case,
        reduce 'config.random_start'
        """
        self.reset()
        if(self.random_start_steps == 0):
            return self.state, 0, 0, self.terminal

        for _ in range(np.random.randint(0, self.random_start_steps)):
            self.step(self.random_step())
        self.render()
        return self.state, 0, self.terminal
    
    def _step(self, action):
        self._state, self.reward, self.terminal, self.info = self.env.step(action)
        
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
    
    def step(self, action):
        self._step(action)
        self.render()
        
        
        
class ALEEnvironment(Environment):
    """
    Environment type to handle ALE environments
    
    states are preprocessed to rgb (self.state)
    to visualise the actual state view self._state
    """
    def __init__(self, config):
        super(ALEEnvironment, self).__init__(config)
        
        screen_width, screen_height, self.action_repeat = \
            config.screen_width, config.screen_height, config.action_repeat
            
        self.dims = (screen_height, screen_width)
        self.is_training = False
        self.info = None
        
    def reset(self):
        """
        Reset the environment
        """
        self._state = self.env.reset()
        self.step(self.random_step())
        self.render()
        
    def preprocess(self, state):
        return imresize(rgb2gray(state)/255., self.dims)
        
    def step(self, action):
        cumulated_reward = 0
        start_lives = self.lives
        
        for _ in range(self.action_repeat):
            self._step(action)
            cumulated_reward = cumulated_reward + self.reward
            
            if(self.is_training and start_lives > self.lives):
                cumulated_reward -= 1
                self.terminal = True
                
            if(self.terminal):
                break
    
        self.reward = cumulated_reward
        self.render()

    @property
    def lives(self):
        if(self.info is None):
            return 0
        else:
            if(not 'ale.lives' in self.info.keys()):
                raise ValueError('Are you sure the requested environment is an'
                                 'ALE environment? Requested:{}'.format(self.env_name))
            return self.info['ale.lives']