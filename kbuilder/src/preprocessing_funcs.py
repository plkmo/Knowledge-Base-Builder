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
import multiprocessing
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def process_text(sent):
    if sent not in [" ", "\n", ""]:
        sent = sent.strip("\n")            
        sent = re.sub('<[A-Z]+/*>', '', sent) # remove special tokens eg. <FIL/>, <S>
        sent = re.sub(r"[\*\"\n\\…\+\-\/\=\(\)‘•€\[\]\|♫:;—”“~`#]", " ", sent)
        sent = re.sub(' {2,}', ' ', sent) # remove extra spaces > 1
        sent = re.sub("^ +", "", sent) # remove space in front
        sent = re.sub(r"([\.\?,!]){2,}", r"\1", sent) # remove multiple puncs
        sent = re.sub(r" +([\.\?,!])", r"\1", sent) # remove extra spaces in front of punc
        #sent = re.sub(r"([A-Z]{2,})", lambda x: x.group(1).capitalize(), sent) # Replace all CAPS with capitalize
        return sent
    return

def process_sent(sent):
    sent = re.sub(r"^ +(.*)", r"\1", sent) # remove space at beginning
    sent = re.sub(' {2,}', ' ', sent)
    sent = sent[0].upper() + sent[1:]
    return sent

def preprocess_corpus(args):
    
    if os.path.isfile("./data/df.pkl"):
        df = load_pickle("./data/df.pkl")
        logger.info("Loaded preprocessed data.")
    else:
        logger.info("Reading file %s" % args.text_file)
        with open(args.text_file, "r", encoding="utf8") as f:
            text = f.readlines()
        
        logger.info("Preprocessing text...")
        cpus = multiprocessing.cpu_count()
        with multiprocessing.Pool(cpus) as pool:
            text1 = list(tqdm(pool.imap(process_text, (sent for sent in text),\
                                        chunksize=int(len(text)//cpus)), total=len(text)))

        logger.info("Splitting into sentences...")
        text = " ".join([t for t in text1 if t is not None])
        del text1
        sents = re.split(r"(?<=[\.\?!])\s", text)
        with multiprocessing.Pool(cpus) as pool:
            sents1 = list(tqdm(pool.imap(process_sent, (sent for sent in sents),\
                                         chunksize=int(len(sents)//cpus)), total=len(sents)))
        
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
    