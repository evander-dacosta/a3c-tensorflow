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
import time

from environment import SimpleEnvironment
from experience import Experience
from agent import BaseAgent
from model import BaseModel

class Config:
    env_name = 'CartPole-v0'
    display =  True
    random_start_steps = 30
    
    h_discount = 0.9
    h_max_steps = 10000
    h_max_q_size = 10000
    
    def create_agent(self):
        pass
    
    
class Model(BaseModel):
    def __init__(self, config, sess):
        self.sess = sess
        self.config = config
        
    def add_summary(self, tag_dict, step):
        pass
        
    def build(self):
        with tf.variable_scope('model'):
            self.target = tf.placeholder(dtype=tf.float32, shape=(None, 2),
                                         name='target')
            self.input = tf.placeholder(dtype=tf.float32, shape=(None, 2),
                                        name='input')
            self.hidden, self.w['hidden_w'], self.w['hidden_b'] = \
                Dense(self.input, output_shape=32, activation_fn='tanh',
                      name='hidden')
            self.output, self.w['output_w'], self.w['output_b'] = \
                Dense(self.hidden, output_shape=2, activation_fn='linear',
                      name='output')
                
        with tf.variable_scope('optimiser'):
            self.optimiser = tf.train.AdamOptimizer()
            self.loss = tf.reduce_mean(self.cost_fn(self.output, self.target))
            self.min_op = self.optimiser.minimize(self.loss)
            
    
    def fit(self, x, y):
        pass
    
    def predict(self, x):
        pass
    
    def cost_fn(self, output, target):
        pass
    
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
            
            

        
