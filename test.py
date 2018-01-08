#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:25:32 2018

@author: evanderdcosta
"""

import gym
import tensorflow as tf
import numpy as np
import multiprocessing as mp
from threading import Thread
import time

from environment import SimpleEnvironment
from experience import Experience
from agent import BaseAgent
from model import BaseModel

class Config:
    env_name = 'CartPole-v0'
    display =  True
    random_start_steps = 30
    
    discount = 0.9
    max_steps = 10000
    max_q_size = 10000
    
    #training hyperparams
    batch_size = 1
    
    def create_agent(self):
        pass
    
    
class Trainer(Thread):
    """
    Base class for Trainer objects 

    What a trainer does:
        1) Collects experience data from an external experience store.
        2) Puts experiences into minibatches
        3) Sends minibatches off to the model to .fit()
        4) Returns the appropriate statistics and logs (? should this be handled by the model?)
    
    The trainer object is a thread daemon. There should only one of its kind
    running. This is an implementation detail which will be handled by the
    server class.
    """
    def __init__(self, server):
        super(Trainer, self).__init__()
        self.setDaemon(True)
    
        self.server = server
        self.config = self.server.config
        
        self.exit_flag = False
        
    def run(self):
        raise NotImplementedError()
        
        
class Predictor(Thread):
    """
    Base class for predictor objects.
    
    What a predictor does:
        1) Collects states from a state store.
        2) Puts it into minibatches
        3) Sends it off to the model to .predict()
        4) Returns the predictions into the wait_q of the appropriate agent
    
    The predictor thread is a thread daemon. There should only one of its kind
    running. This is an implementation detail which will be handled by the
    server class.
    """
    def __init__(self, server):
        super(Trainer, self).__init__()
        self.setDaemon(True)
    
        self.server = server
        self.config = self.server.config
        
        self.exit_flag = False
    
    def run(self):
        raise NotImplementedError()
        
        


        

    
class Model(BaseModel):
    def __init__(self, config, sess):
        self.sess = sess
        self.config = config
        
    def add_summary(self, tag_dict, step):
        pass
        
    def build(self):
        self.parameters = np.random.rand(4) * 2 - 1
        self.noise_scaling = 0.1
            
    
    def fit(self, x, y):
        new_params = self.parameters + (np.random.rand(4) * 2 - 1) * \
                                                self.noise_scaling
        return new_params
    
    def predict(self, x):
        action = 0 if np.matmul(self.parameters, x) < 0 else 1
        return action
    
    def cost_fn(self, output, target):
        pass
    
    
class HillClimbAgent(BaseAgent):
    def predict(self, state):
        self.prediction_q.put((self.id, state))
        #wait for prediction to come back
        a, v = self.wait_q.get()
        v = self.env.reward
        return a, v
    
    

    
class Server:
    def __init__(self, config):
        self.config = config
        self.training_q = mp.Queue(maxsize=config.h_max_q_size)
        self.prediction_q = mp.Queue(maxsize=config.h_max_q_size)
        
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
            
            

        
