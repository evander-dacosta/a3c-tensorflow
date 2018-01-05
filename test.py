#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:25:32 2018

@author: evanderdcosta
"""

import gym
import numpy as np
import multiprocessing as mp
import time

from environment import SimpleEnvironment
from experience import Experience
from agent import BaseAgent

class Config:
    env_name = 'MountainCar-v0'
    display =  False
    random_start_steps = 30
    
    discount = 0.9
    max_steps = 10000
    max_q_size = 10000
    
    
class Server:
    def __init__(self, config):
        self.config = config
        self.training_q = mp.Queue(maxsize=config.maxsize)
        self.prediction_q = mp.Queue(maxsize=config.maxsize)
        
        self.agents = []
    
    def add_agent(self):
        pass
    
    def remove_agent(self):
        pass
    
    def train_model(self):
        pass
    
    def save_model(self):
        pass
    
    def start(self):
        pass
    
    

            
            
            
if __name__ == "__main__":
    predict_q = mp.Queue(10000)
    training_q = mp.Queue(10000)
    agent = BaseAgent(1, predict_q, training_q, Config())
    agent.run()
            
            

        
