#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:44:05 2018

@author: evanderdcosta
"""

from threading import Thread

class Trainer(Thread):
    """
    Base class for Trainer objects 

    What a trainer does:
        1) Collects experience data from an external experience store.
        2) Puts experiences into minibatches
        3) Sends minibatches off to the model to .fit()
        4) Returns the appropriate statistics and logs (? should this be handled by the model?)
    
    The trainer object is a thread daemon. There should only one of its kind
    running. This is an implementation detail which will be handled by the
    server class.
    """
    def __init__(self, server):
        super(Trainer, self).__init__()
        self.setDaemon(True)
    
        self.server = server
        self.config = self.server.config
        
        self.exit_flag = False
        
    def get_training(self):
        if(self.server.is_alive):
            return self.server.training_q.get()
        
    def run(self):
        raise NotImplementedError()