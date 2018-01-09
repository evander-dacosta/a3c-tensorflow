#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:25:32 2018

@author: evanderdcosta
"""

import gym
import tensorflow as tf
import numpy as np
import threading
import time

from environment import SimpleEnvironment, Environment, ALEEnvironment
from experience import Experience
from agent import BaseAgent
from model import BaseModel
from trainer import Trainer
from predictor import Predictor 
from server import Server
from utils import imresize, rgb2gray

class Config:
    env_name = 'Breakout-v0'
    screen_height = 84
    screen_width = 84
    max_reward = 1.
    min_reward = -1.
    cnn_format = 'NHWC'
    
    display =  False
    random_start_steps = 10
    action_repeat = 1
    history_length = 4
    
    discount = 0.9
    max_steps = 10000
    max_q_size = 10000
    memory_size = 1e6
    
    #training hyperparams
    batch_size = 32
    predict_batch_size = 1
    
    
    def create_agent(self):
        pass
    
    

            
            
class ReplayMemory:
    """
    Provides a Queue-like interface to an experience store
    """
    def __init__(self, maxsize):
        self.maxsize = maxsize
        
        self.s = [0] * maxsize
        self.a = ['.'] * maxsize
        self.r = ['.'] * maxsize
        self.s_ = [0] * maxsize
        self.t = ['.'] * maxsize
        
        self.count = 0
        self.put_pointer = 0
        
        self.lock = threading.Lock()
    
    def put(self, experience):
        self.lock.acquire()
        s, a, r, s_, t = experience.get()
        
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s_.append(s_)
        self.t.append(t)
        
        self.count += 1
        self.put_pointer = (self.put_pointer + 1) % self.maxsize
        self.lock.release()

    def get(self):
        """
        Sample a random experience from the experience store
        """
        if(self.count >= self.maxsize):
            idx = np.random.randint(0, self.maxsize)
        else:
            idx = np.random.randint(0, self.put_pointer)
        return self.idx(idx)
    
    def idx(self, idx):
        self.lock.acquire()
        s, a, r, s_, t = self.s[idx], self.a[idx], self.r[idx], self.s_[idx],\
                         self.t[idx]
        payload = (s, a, r, s_, t)
        self.lock.release()
        return payload
    
    
class StateHistory:
    """
    Stores the previous sequence of states that happened according
    to config.history_length.
    
    This is a way for the network to see the immediate sequence preceding 
    the state to leverage sequential information.
    """
    def __init__(self, config):
        self.cnn_format = config.cnn_format
        
        history_length, screen_height, screen_width = \
            config.history_length, config.screen_height, config.screen_width
            
        self.history = np.zeros([history_length, screen_height, screen_width],
                                dtype=np.float32)
        
    def add(self, x):
        self.history[:-1] = self.history[1:]
        self.history[-1] = x
        
    def reset(self):
        self.history *= 0
        
    def get(self):
        if(self.cnn_format == 'NHWC'):
            return np.transpose(self.history, (1, 2, 0))
        else:
            return self.history
    

class DeepQTrainer(Trainer):
    pass


class DeepQPredictor(Predictor):
    pass
    
    
    
class DeepQAgent(BaseAgent):
    def __init__(self, id, prediction_q, training_q, config):
        self.history = StateHistory(config)
        env = ALEEnvironment(config)
        super(DeepQAgent, self).__init__(id, prediction_q, training_q, config,
                                         env)
        
    def predict(self, state):
        #self.prediction_q.put((self.id, state))
        #wait for prediction to come back
        a = self.env.random_step()
        return a

    
    def run_episode(self):
        self.env.reset()
        self.history.add(self.env.state)
        
        random_start_steps = max(self.config.history_length, self.env.random_start_steps)
        for _ in range(random_start_steps):
            self.env.step(self.env.random_step())
            self.history.add(self.env.state)

        t = 0
        while(not self.env.terminal):
            #predict action, value
            prev_state = self.env.state
            action = self.predict(self.history.get())
            self.env.step(action)
            experience = Experience(prev_state, action, self.env.reward, 
                                    self.env.state, self.env.terminal)
            yield experience
            t += 1
            
    def run(self):
        """
        Takes in a game's experience frame-by-frame
        """
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1000 * self.id))

        # Put this in a while loop that checks a shared variable
        # Will keep running episodes until the shared variable reports False
        while(self.exit_flag == 0):
            for experience in self.run_episode():
                self.training_q.put(experience)
                
                
    def simulate(self):
        raise NotImplementedError()


class DeepQServer(Server):
    def __init__(self, config):
        self.config = config
        self.training_q = ReplayMemory(maxsize=config.memory_size)
        self.prediction_q = mp.Queue(maxsize=config.max_q_size)
        
        self.agents = []
        
        self.is_alive = True
        self.start()
        
    def start(self):
        self.trainer = DeepQTrainer()
        self.trainer.start()
        
        self.predictor = DeepQPredictor()
        self.predictor.start()
        
        self.model = DeepQModel(self.config)
        
    def add_agent(self, simulate=False):
        self.agents.append(DeepQAgent(len(self.agents), 
                                          self.prediction_q, self.training_q,
                                          self.config))
        if(simulate):
            self.agents[-1].simulate()
        else:
            self.agents[-1].start()
        
        

        

    

        

if __name__ == "__main__":
    replay = ReplayMemory(100)
    pq = mp.Queue(maxsize=10000)
    agent = DeepQAgent(0, pq, replay, Config())
    agent.start()
    
    

        
