# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:46:33 2019

@author: plkmo
"""
import os
import re
import pandas as pd
import numpy as np
from .preprocessing_funcs import preprocess_corpus
from .utils import save_as_pickle, load_pickle
import spacy
from nltk.stem.porter import PorterStemmer
import networkx as nx
import multiprocessing
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_tagged_data(args):
    if not os.path.isfile("./data/df_tagged.pkl"):
        logger.info("Loading preprocessed text dataframe...")
        df = preprocess_corpus(args)
        logger.info("Done and saved!")
    else:
        logger.info("Loading saved tagged dataset...")
        df = load_pickle("df_tagged.pkl")
        logger.info("Loaded!")
    return df

class Text_KB_Parser(object):
    def __init__(self,):
        self.nlp = spacy.load("en_core_web_lg")
        self.subjects = []
        self.predicates = []
        self.objects = []
        self.triplets = []
        
    def clear_(self):
        self.subjects = []
        self.predicates = []
        self.objects = []
        self.triplets = []
        
    def get_triplets_from_sentence(self, sentence, expand=False):
        triplets = []
        doc = self.nlp(sentence)
        for sent in doc.sents:
            root = sent.root
            subject = None; objs = []
            for child in root.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    subject = child                    
                elif child.dep_ in ["dobj", "attr", "prep", "ccomp"]:
                    objs.append(child)
            
            if (subject is not None) and (len(objs) > 0):
                if not expand:
                    subj = " ".join(str(word) for word in subject.subtree)
                    root_ = str(root)
                    obj_ = " ".join(str(word) for obj in objs for word in obj.subtree)
                    self.subjects.append(subj)
                    self.predicates.append(root_)
                    self.objects.append(obj_)
                    triplets.append((subj, root_, obj_))
                
                else:
                    for obj in objs:
                        subj = " ".join(str(word) for word in subject.subtree)
                        root_ = str(root)
                        obj_ = " ".join(str(word) for word in obj.subtree)
                        self.subjects.append(subj)
                        self.predicates.append(root_)
                        self.objects.append(obj_)
                        triplets.append((subj, root_, obj_))
                    
        self.triplets.extend(triplets)
        return triplets
    
    def cleanup_(self):
        self.subjects = list(set(self.subjects))
        self.predicates = list(set(self.predicates))
        self.objects = list(set(self.objects))
        
    def tag_sentence(self, sentence):
        sent = Sentence(sentence)
        self.ner_tagger.predict(sent)
        self.pos_tagger.predict(sent)
        ner_dict = sent.to_dict(tag_type='ner')
        pos_dict = sent.to_dict(tag_type='pos')
        return ner_dict, pos_dict
    
    def question_parser(self, question):
        triplets = self.get_triplets_from_sentence(question)
        if (len(triplets) > 0):
            triplets = triplets[0]
            subject, predicate, object_ = triplets
            if all(triplets):
                return subject, predicate, object_
        
        # if can't find from dependency tree, crudely get from POS & NER tags
        doc = self.nlp(question)
        entities = doc.ents
        
        selected_entity = entities[0].text if len(entities) > 0 else None
        
        qns = None
        verb = None
        pronouns = ["i", "he", "she", "we", "they",\
                "his", "her", "hers", "it", "its", "you", "your", "yours", "their", "theirs", "them", "mine",\
                "my", "myself", "yourself", "yourselves", "herself", "hiself", "themselves"]
        for word in doc:
            if (word.pos_ in ['PRON', 'ADV']) and (word.text.lower() not in pronouns):
                qns = word.text
            
            if (word.pos_ in ['VERB']) and (word.text.lower() not in ['is',]):
                if verb is None:
                    verb = word.text
                else:
                    if (word.text.lower() not in ['does', 'do', 'did', 'have', 'has', 'had']):
                        verb = word.text
            
            if (selected_entity == None) and (word.pos_ in ["X", "NOUN", "PROPN"]):
                selected_entity = word.text
        
        if verb is None:
            verb = "is"
        
        if (qns is not None) and (verb is not None) and (selected_entity is not None):
            qns = re.sub("[\.\?,!\"':;><]+", "", qns)
            verb = re.sub("[\.\?,!\"':;><]+", "", verb)
            selected_entity = re.sub("[\.\?,!\"':;><]+", "", selected_entity)
        return qns, verb, selected_entity
        
def search(query, triplets, stemmer, key=0):
    '''
    Searches triplets for query term. key = 0 (searches in subjects); 1 (searches in predicates); 2 (searches in objects)
    '''
    matched_triplets = []
    term = r"\b%s\b" % query
    term = term.lower()
    stemmed_term = stemmer.stem(term)
    logger.info("Searching and collecting results...")
    for tup in tqdm(triplets):    
        target = tup[key].lower()
        stemmed_target = " ".join([stemmer.stem(t) for t in target.split()])
        if re.search(term, target) is not None:
            matched_triplets.append(tup)
        elif re.search(stemmed_term, stemmed_target) is not None:
            matched_triplets.append(tup)
    return matched_triplets

def matcher(tup, key, term, stemmer, stemmed_term):
    matched_triplets_tmp = []
    target = tup[key].lower()
    stemmed_target = " ".join([stemmer.stem(t) for t in target.split()])
    if re.search(term, target) is not None:
        matched_triplets_tmp.append(tup)
    elif re.search(stemmed_term, stemmed_target) is not None:
        matched_triplets_tmp.append(tup)
    return matched_triplets_tmp

def wrapped_matcher(args):
    if isinstance(args[1], int) and isinstance(args[2], str):
        matched_triplets_tmp = matcher(*args)
    else:
        matched_triplets_tmp = double_matcher(*args)
    return matched_triplets_tmp

def parallel_search(query, triplets, stemmer, key=0):
    matched_triplets = []
    if isinstance(query, str) and isinstance(key, int):
        term = r"\b%s\b" % query
        term = term.lower()
        stemmed_term = stemmer.stem(term)
    else:
        term = [r"\b%s\b" % q for q in query]
        term = [t.lower() for t in term]
        stemmed_term = [stemmer.stem(t) for t in term]
    
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpus)
    
    args_list = [(tup, key, term, stemmer, stemmed_term) for tup in triplets]
    logger.info("Searching...")
    results = pool.map(wrapped_matcher, args_list)
    
    logger.info("Collecting results...")
    for result in tqdm(results):
        matched_triplets.extend(result)
    
    return matched_triplets

def double_matcher(tup, keys, terms, stemmer, stemmed_terms): # keys = 2 iterables
    matched_triplets_tmp = []
    target = tup[keys[0]].lower(); target2 = tup[keys[1]].lower()
    stemmed_target = " ".join([stemmer.stem(t) for t in target.split()])
    stemmed_target2 = " ".join([stemmer.stem(t) for t in target2.split()])
    if (re.search(terms[0], target) is not None) and (re.search(terms[1], target2) is not None):
        matched_triplets_tmp.append(tup)
    elif (re.search(stemmed_terms[0], stemmed_target) is not None) and (re.search(stemmed_terms[1], stemmed_target2) is not None):
        matched_triplets_tmp.append(tup)
    return matched_triplets_tmp

def answer(qns, verb, selected_entity, triplets, stemmer):
    if (qns == None) or (verb == None) or (selected_entity == None):
        return "I can't find what you want. Try something else."
    
    best_result = parallel_search([selected_entity, verb], triplets, stemmer, key=[0,1])
    if len(best_result) == 0:
        matched_subjects = parallel_search(selected_entity, triplets, stemmer, key=0)
        if len(matched_subjects) == 0:
            return "I can't find what you want. Try something else."
        else:
            choice = np.random.choice([i for i in range(len(matched_subjects))])
            ans = " ".join(matched_subjects[choice]) + "."
            ans = ans[0].upper() + ans[1:]
            return ans
    else:
        choice = np.random.choice([i for i in range(len(best_result))])
        ans = " ".join(best_result[choice]) + "."
        ans = ans[0].upper() + ans[1:]
        return ans
    
class KB_Bot(object):
    def __init__(self, args=None):
        if args is None:
            args = load_pickle("args.pkl")
        self.df = load_tagged_data(args)
        self.text_parser = Text_KB_Parser()
        
        if os.path.isfile("./data/subjects.pkl") and os.path.isfile("./data/predicates.pkl"):
            self.subjects = load_pickle("subjects.pkl")
            self.predicates = load_pickle("predicates.pkl")
            self.objects = load_pickle("objects.pkl")
            self.triplets = load_pickle("triplets.pkl")
            logger.info("Loaded KB from saved files.")
        else:
            logger.info("Extracting KB...")
            
            self.df.progress_apply(lambda x: self.text_parser.get_triplets_from_sentence(x['sents']), axis=1)
            self.text_parser.cleanup_()
            self.subjects = self.text_parser.subjects
            self.predicates = self.text_parser.predicates
            self.objects = self.text_parser.objects
            self.triplets = self.text_parser.triplets
            self.text_parser.clear_()
            save_as_pickle("subjects.pkl", self.subjects)
            save_as_pickle("predicates.pkl", self.predicates)
            save_as_pickle("objects.pkl", self.objects)
            save_as_pickle("triplets.pkl", self.triplets)
            logger.info("Done and saved!")
        
        logger.info("%d relationships in the KB." % len(self.triplets))
        self.stemmer = PorterStemmer()
        
    def query(self, term, key=0):
        #key_dict = {'subject':0, 'predicate':1, 'object':2}
        matched_triplets = parallel_search(term, self.triplets, self.stemmer, key=key)
        print("%d results found." % len(matched_triplets))
        return matched_triplets
    
    def ask(self, question):
        qns, verb, selected_entity = self.text_parser.question_parser(str(question))
        print("\n\
              ***Identified Subject: %s\n\
              ***Identified Predicate: %s\n\
              ***Identified Object: %s\n" % (qns, verb, selected_entity))
        ans = answer(qns, verb, selected_entity, self.triplets, self.stemmer)
        print(ans)
        return ans
        
    def chat(self,):
        while True:
            user_input = input("Type your query:\n")
            if user_input in ["exit", "quit"]:
                break
            ans = self.ask(user_input)
        return ans
    
    
if __name__ == "__main__":
    bot = KB_Bot()
    bot.chat()

    '''
    ### Graphify ###
    logger.info("Initializing graph...")
    G = nx.MultiDiGraph()
    logger.info("Adding nodes...")
    G.add_nodes_from(subjects)
    G.add_nodes_from(objects)
    logger.info("Building edges...")
    edges = [(sub, obj, {"attr":predicate}) for sub, predicate, obj in tqdm(triplets)]
    G.add_edges_from(edges)
    '''