#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:44:34 2018

@author: evanderdcosta
"""

from threading import Thread

class Predictor(Thread):
    """
    Base class for predictor objects.
    
    What a predictor does:
        1) Collects states from a state store.
        2) Puts it into minibatches
        3) Sends it off to the model to .predict()
        4) Returns the predictions into the wait_q of the appropriate agent
    
    The predictor thread is a thread daemon. There should only one of its kind
    running. This is an implementation detail which will be handled by the
    server class.
    """
    def __init__(self, server):
        super(Predictor, self).__init__()
        self.setDaemon(True)
    
        self.server = server
        self.config = self.server.config
        
        self.exit_flag = False
        
    def get_prediction(self):
        if(self.server.is_alive):
            return self.server.prediction_q.get()
    
    def run(self):
        raise NotImplementedError()