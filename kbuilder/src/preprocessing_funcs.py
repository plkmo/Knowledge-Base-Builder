#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 09:20:29 2019

@author: tsd
"""

import os
import re
import pandas as pd
from .utils import save_as_pickle, load_pickle
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def preprocess_corpus(args):
    
    if os.path.isfile("./data/df.pkl"):
        df = load_pickle("./data/df.pkl")
        logger.info("Loaded preprocessed data.")
    else:
        logger.info("Reading file %s" % args.text_file)
        with open(args.text_file, "r", encoding="utf8") as f:
            text = f.readlines()
        
        text1 = []
        for sent in tqdm(text, total=len(text)):
            if sent not in [" ", "\n"]:
                text1.append(sent.strip("\n"))
        
        logger.info("Preprocessing text...")
        text = " ".join(text1)
        text = re.sub('<[A-Z]+/*>', '', text) # remove special tokens eg. <FIL/>, <S>
        text = re.sub(r"[\*\"\n\\…\+\-\/\=\(\)‘•€\[\]\|♫:;—”“~`#]", " ", text)
        text = re.sub(' {2,}', ' ', text) # remove extra spaces > 1
        text = re.sub("^ +", "", text) # remove space in front
        text = re.sub(r"([\.\?,!]){2,}", r"\1", text) # remove multiple puncs
        text = re.sub(r" +([\.\?,!])", r"\1", text) # remove extra spaces in front of punc
        text = re.sub(r"([A-Z]{2,})", lambda x: x.group(1).capitalize(), text) # Replace all CAPS with capitalize
        del text1
        
        sents = re.split("\.", text)
        sents1 = []
        for sent in sents:
            sent = re.sub(r"^ +(.*)", r"\1", sent) # remove space at beginning
            sent = re.sub(' {2,}', ' ', sent + ".")
            sent = sent[0].upper() + sent[1:]
            sents1.append(sent)
        sents = sents1; del sents1
        df = pd.DataFrame(data={"sents":sents})
        df['length'] = df.progress_apply(lambda x: len(x['sents']), axis=1)
        
        logger.info("Removing char sequences of length < 7")
        df = df[df['length'] >= 7]
        
        save_as_pickle("./data/df.pkl", df)
        logger.info("Done and saved.")
        
    return df
    
if __name__ == "__main__":
    df = preprocess_corpus()
    