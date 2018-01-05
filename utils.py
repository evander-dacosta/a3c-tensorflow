#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:25:27 2018

@author: evanderdcosta
"""

import inspect

def class_vars(obj):
    """
    Collects all hyperparameters from the config file
    we know a hyperparam b/c it's preceded by a 'h_'
    e.g. h_optimizer
    """
    return{k:v for k,v in inspect.getmembers(obj) 
        if k.startswith('h_') and not callable(k)}