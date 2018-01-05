#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:25:32 2018

@author: evanderdcosta
"""

import gym
import numpy as np

from environment import SimpleEnvironment
from experience import Experience

class Config:
    env_name = 'CartPole-v0'
    display = True
    
    random_start = 30
    
    
    

        

        
    
    
    
    

env = SimpleEnvironment(Config())
env.reset()

while(not env.terminal):
    env.random_step()
    env.render()
    print(env.reward)