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
    
    
class BaseAgent(mp.Process):
    def __init__(self, id, prediction_q, training_q, config):
        super(BaseAgent, self).__init__()
        self.config = config
        
        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        
        self.env = SimpleEnvironment(self.config)
        self.n_actions = self.env.action_size
        self.actions = np.arange(self.n_actions)
        
        self.discount, self.max_steps = config.discount, config.max_steps
        # One frame at a time
        self.wait_q = mp.Queue(maxsize=1)
        
        
    def predict(self, state):
        self.prediction_q.put((self.id, state))
        #wait for prediction to come back
        #a, v = self.wait_q.get()
        a = self.env.random_step()
        v = self.env.reward
        return a, v

    
    def run_episode(self):
        self.env.reset()
        self.env.random_start()
        t = 0
        experiences = []
        
        
        while(not self.env.terminal):
            #predict action, value
            action, value = self.predict(self.env.state)
            self.env.step(action)
            self.env.render()
            
            experience = Experience(action, self.env.state, self.env.reward,
                                    self.env.terminal)
            experiences.append(experience)
            yield experience
            t += 1
            
    def run(self):
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1000 * self.id))
        
        # Put this in a while loop that checks a shared variable
        # Will keep running episodes until the shared variable reports False
        for experience in self.run_episode():
            print(experience.state, experience.reward)
            self.training_q.put(experience)
            
            
            
if __name__ == "__main__":
    predict_q = mp.Queue(100)
    training_q = mp.Queue(100)
    agent = BaseAgent(1, predict_q, training_q, Config())
    agent.run()
            
            

        
