#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:25:27 2018

@author: evanderdcosta
"""

import inspect
import numpy as np
from scipy.misc import imresize

def class_vars(obj):
    """
    Collects all hyperparameters from the config file
    we know a hyperparam b/c it's preceded by a 'h_'
    e.g. h_optimizer
    """
    return{k:v for k,v in inspect.getmembers(obj) 
        if k.startswith('h_') and not callable(k)}
    
    
def rgb2gray(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])