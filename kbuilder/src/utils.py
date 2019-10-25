# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:09:18 2019

@author: WT
"""
import os
import pickle
import numpy as np

def load_pickle(filename):
    completeName = filename
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = filename
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

class Config(object):
    def __init__(self,):
        self.text_file = './data/text.txt'
        self.state_dict= './data/KB_Bot_state_dict.pkl'