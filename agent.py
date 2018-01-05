#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:25:53 2018

@author: evanderdcosta
"""

import multiprocessing as mp
import numpy as np
import time
from environment import SimpleEnvironment
from experience import Experience

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