#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:25:32 2018

@author: evanderdcosta
"""

import gym
import tensorflow as tf
import numpy as np

import time

from environment import SimpleEnvironment
from experience import Experience
from agent import BaseAgent
from model import BaseModel
from trainer import Trainer
from predictor import Predictor 
from server import Server

class Config:
    env_name = 'CartPole-v0'
    display =  False
    random_start_steps = 10
    
    discount = 0.9
    max_steps = 10000
    max_q_size = 10000
    
    #training hyperparams
    batch_size = 32
    predict_batch_size = 1
    
    def create_agent(self):
        pass
        

class HillClimbModel(BaseModel):
    def __init__(self, config, sess):
        super(HillClimbModel, self).__init__(config, sess)
        
        # Keep track of the best parameters and its score
        self.best_params = None
        self.best_params_score = -np.inf
        
    def add_summary(self, tag_dict, step):
        pass
        
    def build(self):
        self.noise_scaling = 0.1
                
        self.parameters = self.generate_params()
        self.best_params = self.parameters[:]


    def fit(self, current_scores):
        print("Model: Current score: {}, Best score: {}".format(current_scores, 
                                                                  self.best_params_score))
        if(current_scores > self.best_params_score):
            self.best_params_score = current_scores
            self.best_params = self.parameters
        self.parameters = self.generate_params(self.parameters)
    
    def predict(self, x):
        action = 0 if np.matmul(self.parameters, x) < 0 else 1
        return action
    
    def cost_fn(self, output, target):
        pass
    
    def generate_params(self, params=None):
        if(params is None):
            params = np.random.rand(self.state_shape) * 2 - 1
        return params + (np.random.rand(self.state_shape) * 2 - 1) * \
                                                            self.noise_scaling
                                                            
    
    
class HillClimbAgent(BaseAgent):
    def predict(self, state):
        self.prediction_q.put((self.id, state))
        #wait for prediction to come back
        a = self.wait_q.get()
        v = self.env.reward
        return a, v
    
    def run(self):
        np.random.seed(np.int32(time.time() % 1000 * self.id))

        while(self.exit_flag.value == 0):
            total_reward = 0
            for experience in self.run_episode():
                total_reward += experience.reward
                if(experience.terminal):
                    self.training_q.put((self.id, total_reward))
                    #total_reward = 0.
                
                
class HillClimbTrainer(Trainer):
    def run(self):
        while(not self.exit_flag):
            batch_size = 0
            all_rewards = []
            while(batch_size <= self.config.batch_size):
                id, total_reward = self.get_training()
                all_rewards.append(total_reward)
                batch_size += 1
            self.server.train_model(np.mean(all_rewards))
            
            
class HillClimbPredictor(Predictor):
    def run(self):
        while(not self.exit_flag):
            id, state = self.get_prediction()
            result = self.server.predict_model(state)
            self.server.agents[id].wait_q.put(result)
                
                
class HillClimbServer(Server):
    def __init__(self, config):
        super(HillClimbServer, self).__init__(config)
        
    
    def start(self):
        self.trainer = HillClimbTrainer(self)
        self.trainer.start()
        
        self.predictor = HillClimbPredictor(self)
        self.predictor.start()
        
        self.model = HillClimbModel(self.config, None)
        
        
    def add_agent(self, simulate=False):
        self.agents.append(HillClimbAgent(len(self.agents), 
                                          self.prediction_q, self.training_q,
                                          self.config))
        if(simulate):
            self.agents[-1].simulate()
        else:
            self.agents[-1].start()
        
    def remove_agent(self):
        self.agents[-1].exit_flag.value = True
        self.agents[-1].terminate()
        self.agents.pop()
        print("Server: Agent stopped")
        
    def train_model(self, scores):
        self.model.fit(scores)
        
    def predict_model(self, states):
        return self.model.predict(states)
    
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
        

if __name__ == "__main__":
    server = HillClimbServer(Config())
    server.add_agent()
            
            

        
