#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:31:00 2019

@author: weetee
"""
from kbuilder.src.preprocessing_funcs import preprocess_corpus
from kbuilder.src.KB_funcs import KB_Bot
from kbuilder.src.utils import save_as_pickle
import logging
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--text_file', type=str, default='./data/text.txt', help="Input text file")
    parser.add_argument('--state_dict', type=str, default="./data/KB_Bot_state_dict.pkl", \
                        help="Saved state dict for KB_Bot")
    args = parser.parse_args()
    save_as_pickle('./data/args.pkl', args)
    
    bot = KB_Bot()
    bot.chat()
