#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 14:54:29 2018

@author: evander
"""

import tensorflow as tf

activation_fns = {'sigmoid': tf.nn.sigmoid,
                  'softmax': tf.nn.softmax,
                  'relu': tf.nn.relu,
                  'tanh': tf.nn.tanh,
                  'linear': lambda x: x
                  }

def Dense(x, output_size, w=None, activation_fn=tf.nn.relu, 
          name='linear'):
    shape = x.get_shape().as_list()
    
    if(isinstance(activation_fn, str)):
        if(activation_fn in activation_fns.keys()):
            activation_fn = activation_fns[activation_fn]
        else:
            raise ValueError('Unknown activation '
                             'function: {}'.format(activation_fn))
        
    
    with tf.variable_scope(name):
        if(w is None):
            w = tf.get_variable('weight_matrix', [shape[1], output_size], 
                tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        
        b = tf.get_variable('bias', [output_size,], 
                            initializer=tf.constant_initializer(0.))
        out = tf.nn.bias_add(tf.matmul(x, w), b)
        
        if(activation_fn != None):
            return activation_fn(out), w, b
        else:
            return out, w, b