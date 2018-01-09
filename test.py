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
from queue import Queue
from functools import reduce

from environment import SimpleEnvironment, Environment, ALEEnvironment
from experience import Experience
from agent import BaseAgent
from model import BaseModel
from trainer import Trainer
from predictor import Predictor 
from server import Server
from utils import imresize, rgb2gray
from ops import linear, conv2d, clipped_error

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
    memory_size = 10000
    
    #training hyperparams
    batch_size = 32
    predict_batch_size = 1
    learning_rate = 0.00025
    learning_rate_minimum = 0.00025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 5000

    
    

            
            
class ReplayMemory:
    """
    Provides a Queue-like interface to an experience store
    """
    def __init__(self, maxsize, config):
        self.maxsize = maxsize
        
        self.s = [0] * maxsize
        self.a = ['.'] * maxsize
        self.r = ['.'] * maxsize
        self.s_ = [0] * maxsize
        self.t = ['.'] * maxsize
        
        self.count = 0
        self.put_pointer = 0
        
        self.lock = threading.Lock()
        
        self.history_length = config.history_length
        self.batch_size = config.batch_size
        self.cnn_format = config.cnn_format
    
    def put(self, experience):
        self.lock.acquire()
        s, a, r, s_, t = experience.get()
        
        self.s[self.put_pointer] = s
        self.a[self.put_pointer] = a
        self.r[self.put_pointer] = r
        self.s_[self.put_pointer] = s_
        self.t[self.put_pointer] = t
        
        self.count += 1
        self.put_pointer = (self.put_pointer + 1) % self.maxsize
        self.lock.release()

    def get(self):
        """
        Sample a random experience from the experience store
        """
        if(self.count >= self.maxsize):
            idx = np.random.randint(self.history_length, self.maxsize)
        else:
            idx = np.random.randint(self.history_length, self.put_pointer)
        return self.idx(idx)
    
    def idx(self, idx):
        self.lock.acquire()

        idx = idx % min(self.count, self.maxsize)
        if(idx >= self.history_length - 1):
            payload = [self.s[(idx - (self.history_length - 1)):(idx+1)],
                    self.a[idx],
                    self.r[idx],
                    np.array(self.s_[(idx - (self.history_length - 1)):(idx+1)]),
                    self.t[idx]]
        else:
            indices = [(idx - i) % self.count for i in reversed(range(self.history_length))]
            payload = [np.array(self.s[indices]), self.a[idx], self.r, 
                       np.array(self.s_[idx]), self.t]
            
        self.lock.release()
        return payload

    
    def sample_batch(self):
        # Hangs the trainer until we put some more 
        # experience into the memory
        while(self.count < self.batch_size):
            continue
        
        indices = []
        s = [0] * self.batch_size
        a = [0] * self.batch_size
        r = [0] * self.batch_size
        s_ = [0] * self.batch_size
        t = [0] * self.batch_size
        max_count = min(self.maxsize, self.count)
        while(len(indices) < self.batch_size):
            while True:
                index = np.random.randint(self.history_length, max_count)
                if(index >= self.put_pointer and index - self.history_length < self.put_pointer):
                    continue
                if(np.any(self.t[(index - self.history_length):index])):
                    continue
                indices.append(index)
                break
        
        for i, idx in enumerate(indices):
            payload = self.idx(idx)
            s[i] = payload[0]
            a[i] = payload[1]
            r[i] = payload[2]
            s_[i] = payload[3]
            t[i] = payload[4]
            
        s = np.array(s)
        s_ = np.array(s_)
        
        if(self.cnn_format == 'NHWC'):
            s = np.transpose(s, (0, 2, 3, 1))
            s_ = np.transpose(s_, (0, 2, 3, 1))
        return s, a, r, s_, t
    
    
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
    def run(self):
        while(not self.exit_flag):
            s, a, r, s_, t = self.server.training_q.sample_batch()
            print(self.server.model.fit(s, a, r, s_, t))


class DeepQPredictor(Predictor):
    def run(self):
        while(not self.exit_flag):
            ids = []
            batch = []
            while(len(batch) < self.server.config.predict_batch_size):
                id, state = self.get_prediction()
                ids.append(id)
                batch.append(state)
                
            results = self.server.model.predict(np.array(batch))
            for i, id in enumerate(ids):
                self.server.agents[id].wait_q.put(results[i])




class DeepQModel(BaseModel):
    def __init__(self, config, sess):
        self.screen_height, self.screen_width, self.history_length = \
            config.screen_height, config.screen_width, config.history_length
            
        self.cnn_format = config.cnn_format
        
        self.learning_rate, self.learning_rate_minimum, self.learning_rate_decay = \
            config.learning_rate, config.learning_rate_minimum, config.learning_rate_decay
            
        self.learning_rate_decay_step = config.learning_rate_decay_step
        
        self.discount = config.discount
        
        super(DeepQModel, self).__init__(config, sess)
        
        self.step = 0
        

    def build(self):
        self.w = {}
        self.t_w = {}
        
        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu
        
        with tf.variable_scope('prediction'):
            if(self.cnn_format == 'NHWC'):
                self.s_t = tf.placeholder('float32',
                    [None, self.screen_height, self.screen_width,
                     self.history_length], name='s_t')
            else:
                self.s_t = tf.placeholder('float32',
                    [None, self.history_length, self.screen_height,
                     self.screen_width], name='s_t')
            
            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
                           32, [8, 8], [4, 4], initializer, activation_fn, 
                           self.cnn_format, name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
                           64, [4, 4], [2, 2], initializer, activation_fn, 
                           self.cnn_format, name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
                           64, [3, 3], [1, 1], initializer, activation_fn, 
                           self.cnn_format, name='l3')
            
            shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, 
                                      [-1, reduce(lambda x, y: x * y, shape[1:])])
            
            self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 
                                    512, activation_fn=activation_fn, name='l4')
            self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, 
                                    self.action_size, name='q')
            
            self.q_action = tf.argmax(self.q, dimension=1)
            

            
        with tf.variable_scope('target'):
            if(self.cnn_format == 'NHWC'):
                self.target_s_t = tf.placeholder('float32',
                    [None, self.screen_height, self.screen_width,
                     self.history_length], name='target_s_t')
            else:
                self.target_s_t = tf.placeholder('float32',
                    [None, self.history_length, self.screen_height,
                     self.screen_width], name='target_s_t')
    
            self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t, 
                             32, [8, 8], [4, 4], initializer, activation_fn, 
                             self.cnn_format, name='target_l1')
            self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1,
                             64, [4, 4], [2, 2], initializer, activation_fn, 
                             self.cnn_format, name='target_l2')
            self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2,
                             64, [3, 3], [1, 1], initializer, activation_fn, 
                             self.cnn_format, name='target_l3')
            
            shape = self.target_l3.get_shape().as_list()
            self.target_l3_flat = tf.reshape(self.target_l3,
                                             [-1, reduce(lambda x, y: x * y, shape[1:])])
            
            self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
                    linear(self.target_l3_flat, 512, 
                           activation_fn=activation_fn, name='target_l4')
            self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
                    linear(self.target_l4, self.action_size, name='target_q')
                    
            self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)
            
        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}
            
            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(),
                                                      name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])
                
        with tf.variable_scope('optimiser'):
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')
            
            action_one_hot = tf.one_hot(self.action, self.action_size, 1., 0., name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')
            
            self.delta = self.target_q_t - q_acted
            
            self.global_step = tf.Variable(0, trainable=False)
            
            self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                tf.train.exponential_decay(
                        self.learning_rate,
                        self.learning_rate_step,
                        self.learning_rate_decay_step,
                        self.learning_rate_decay,
                        staircase=True))
            self.optim = tf.train.RMSPropOptimizer(
                    self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)
            
        self.sess.run(tf.global_variables_initializer())
        #self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)
        
        #self.load_model()
        self.update_target_q_network()
                
    def update_target_q_network(self):
        for name in self.w.keys():
            self.sess.run(self.t_w_assign_op[name], {
                    self.t_w_input[name]: self.w[name].eval(session=self.sess)})
        
    def fit(self, s_t, action, reward, s_, terminal):
        
        q_t_plus_1 = self.sess.run(self.target_q, {self.target_s_t: s_})
        
        terminal = np.array(terminal) + 0.
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
        target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward
        
        
        _, q_t, loss = self.sess.run([self.optim, self.q, self.loss], {
                self.target_q_t: target_q_t,
                self.action: action,
                self.s_t: s_t,
                self.learning_rate_step: self.step})
        self.step += 1
        return loss
        
    def predict(self, x):
        return self.sess.run(self.q_action, {self.s_t: x})
        
    def add_summary(self, summary_tags):
        raise NotImplementedError()    
    
    
class DeepQAgent(BaseAgent):
    def __init__(self, id, prediction_q, training_q, config):
        self.history = StateHistory(config)
        env = ALEEnvironment(config)
        super(DeepQAgent, self).__init__(id, prediction_q, training_q, config,
                                         env)
        
    def predict(self, state):
        self.prediction_q.put((self.id, state))
        #wait for prediction to come back
        a = self.wait_q.get()
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
        self.training_q = ReplayMemory(maxsize=config.memory_size,
                                       config=config)
        self.prediction_q = Queue(maxsize=config.max_q_size)
        
        self.agents = []
        
        self.is_alive = True
        
    def start(self):
        tf.reset_default_graph()
        
        self.trainer = DeepQTrainer(self)
        self.trainer.start()
        
        self.predictor = DeepQPredictor(self)
        self.predictor.start()
        
        self.session = tf.Session()
        self.model = DeepQModel(self.config, self.session)
        self.model.build()
        
        self.add_agent()
        
    def add_agent(self, simulate=False):
        self.agents.append(DeepQAgent(len(self.agents), 
                                          self.prediction_q, self.training_q,
                                          self.config))
        if(simulate):
            self.agents[-1].simulate()
        else:
            self.agents[-1].start()
        
        

        

    

        

if __name__ == "__main__":
    server = DeepQServer(Config())
    server.start()
    
    

        
