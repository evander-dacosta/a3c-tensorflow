#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:24:39 2018

@author: evanderdcosta
"""


import tensorflow as tf
import os
import gym
from utils import class_vars
    

class BaseModel(object):
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        self._saver = None
        
        try:
            self._attrs = config.__dict__['__flags']
        except:
            self._attrs = class_vars(config)
        print(self._attrs)
        
        for attr in self._attrs:
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(self.config, attr))
            
        # Models have a local copy of an environment, but NEVER! use it
        self._env = gym.make(config.env_name)
        self._example_state = None
            
        
    def save_model(self, step=None):
        print("[*] Saving a checkpoint")
        if(not os.path.exists(self.checkpoint_dir)):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, self.checkpoint_dir, global_step=self.step)
        
    def load_model(self):
        print("[*] Loading a model")
        chkpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if(chkpt and chkpt.model_checkpoint_path):
            chkpt_name = os.path.basename(chkpt.model_checkpoint_path)
            fname = os.path.join(self.checkpoint_dir, chkpt_name)
            self.saver.restore(self.sess, fname)
            print("[*] SUCCESS!")
            return True
        else:
            print("Model load failed....")
            return False
        
    @property
    def action_size(self):
        return self._env.action_space.n
    
    @property
    def state_shape(self):
        if(self._example_state is None):
            self._example_state = self._env.reset()
        return len(self._example_state)
        
    @property
    def checkpoint_dir(self):
        return os.path.join('checkpoints', self.model_dir)
    
    @property
    def model_dir(self):
        model_dir = self.config.name
        for k, v in self._attrs.items():
            if not k.startswith('_') and k not in ['display']:
                model_dir += "/%s-%s" % (k, ",".join([str(i) for i in v])
                         if type(v) == list else v)
        return model_dir + '/'

    @property
    def saver(self):
        if(self._saver == None):
            self._saver = tf.train.Saver(max_to_keep=10)
        return self._saver
    
    def build(self):
        raise NotImplementedError()
        
        
    def fit(self, x, y):
        raise NotImplementedError()
        
    def predict(self, x):
        raise NotImplementedError()
        
    def add_summary(self, summary_tags):
        raise NotImplementedError()