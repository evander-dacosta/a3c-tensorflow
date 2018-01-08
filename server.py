#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:45:03 2018

@author: evanderdcosta
"""

import multiprocessing as mp

class Server:
    def __init__(self, config):
        self.config = config
        self.training_q = mp.Queue(maxsize=config.max_q_size)
        self.prediction_q = mp.Queue(maxsize=config.max_q_size)
        
        self.agents = []
        
        self.is_alive = True
        self.start()

        
    def build(self):
        pass
    
    def add_agent(self):
        pass
    
    def remove_agent(self):
        pass
    
    def train_model(self):
        pass
    
    def predict_model(self):
        pass
    
    def save_model(self):
        pass
    
    def stop(self):
        pass
    
    def start(self):
        pass
    
    def simulate(self):
        pass