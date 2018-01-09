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
        
    def remove_agent(self):
        self.agents[-1].exit_flag.value = True
        self.agents[-1].terminate()
        self.agents.pop()
        print("Server: Agent stopped")
    
    def train_model(self, scores):
        self.model.fit(scores)
        
    def predict_model(self, states):
        return self.model.predict(states)
    
    def save_model(self):
        self.model.save_model()
    
    def stop(self):
        self.trainer.exit_flag = True
        self.trainer.join()
        print("Server: Trainer terminated")
        
        self.predictor.exit_flag = True
        self.predictor.join()
        print("Server: Predictor terminated")
        
        for i in range(len(self.agents)):
            self.remove_agent()
        print("Server: Agents terminated")
        self.is_alive = False
    
    def start(self):
        raise NotImplementedError()
    
    def add_agent(self, simulate=False):
        """self.agents.append(HillClimbAgent(len(self.agents), 
                                          self.prediction_q, self.training_q,
                                          self.config))
        if(simulate):
            self.agents[-1].simulate()
        else:
            self.agents[-1].start()"""
        raise NotImplementedError()