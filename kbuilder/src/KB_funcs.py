# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:46:33 2019

@author: plkmo
"""
import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
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
        df = preprocess_corpus(args)
    else:
        logger.info("Loading saved tagged dataset...")
        df = load_pickle("./data/df_tagged.pkl")
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
        try:
            self.subject_entities = []; self.object_entities = []
            self.subject_entities_d = defaultdict(list); self.object_entities_d = defaultdict(list)
        except:
            pass
        
    def get_triplets_from_sentence(self, sentence, expand=False, store=False):
        triplets = []
        doc = self.nlp(sentence)
        for sent in doc.sents:
            root = sent.root
            subject = None; objs = []
            for child in root.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    if len(re.findall("[a-z]+",child.text.lower())) > 0: # filter out all numbers/symbols
                        subject = child                    
                elif child.dep_ in ["dobj", "attr", "prep", "ccomp"]:
                    objs.append(child)
            
            if (subject is not None) and (len(objs) > 0):
                if not expand:
                    subj = " ".join(str(word) for word in subject.subtree)
                    root_ = str(root)
                    obj_ = " ".join(str(word) for obj in objs for word in obj.subtree)
                    if store:
                        self.subjects.append(subj)
                        self.predicates.append(root_)
                        self.objects.append(obj_)
                    triplets.append((subj, root_, obj_))
                
                else:
                    for obj in objs:
                        subj = " ".join(str(word) for word in subject.subtree)
                        root_ = str(root)
                        obj_ = " ".join(str(word) for word in obj.subtree)
                        if store:
                            self.subjects.append(subj)
                            self.predicates.append(root_)
                            self.objects.append(obj_)
                        triplets.append((subj, root_, obj_))
        
        if store:            
            self.triplets.extend(triplets)
        return triplets
    
    def cleanup_func_(self, triplet):
        subject_entities = []; object_entities = []
        subject_entities_d = defaultdict(list); object_entities_d = defaultdict(list)
        s, p, o = triplet
        s_doc = self.nlp(s)
        s_ents = s_doc.ents
        if len(s_ents) > 0:
            s_ents = [se.text for se in s_ents if len(re.findall("[a-z]+", se.text.lower())) > 0]
            subject_entities.extend(s_ents)
            for se in s_ents:
                subject_entities_d[se].append(s)
        
        o_doc = self.nlp(o)
        o_ents = o_doc.ents
        if len(o_ents) > 0:
            o_ents = [oe.text for oe in o_ents if len(re.findall("[a-z]+", oe.text.lower())) > 0]
            object_entities.extend(o_ents)
            for oe in o_ents:
                object_entities_d[oe].append(o)
                
        return subject_entities, object_entities, subject_entities_d, object_entities_d 
    
    def cleanup_(self):
        logger.info("Removing duplicates...")
        self.subjects = list(set(self.subjects))
        self.predicates = list(set(self.predicates))
        self.objects = list(set(self.objects))
        self.triplets = list(set(self.triplets))
        logger.info("Done!")
        
        logger.info("Merging entities...")
        logger.info("Collecting entities in subjects and objects...")
        cpus = multiprocessing.cpu_count()
        with multiprocessing.Pool(cpus) as pool:
            result = list(tqdm(pool.imap(self.cleanup_func_, (triplet for triplet in self.triplets),\
                                         chunksize=int(len(self.triplets)//cpus)),\
                               total=len(self.triplets)))
        
        logger.info("Collecting results...")
        self.subject_entities = []; self.object_entities = []
        self.subject_entities_d = defaultdict(list); self.object_entities_d = defaultdict(list)
        for se, oe, sed, oed in tqdm(result, total=len(result)):
            self.subject_entities.extend(se); self.object_entities.extend(oe)
            for k, v in sed.items():
                self.subject_entities_d[k].extend(v)
            for k, v in oed.items():
                self.object_entities_d[k].extend(v)
            
        self.subject_entities = list(set(self.subject_entities))
        self.object_entities = list(set(self.object_entities))
        
        logger.info("Done!")
        
    def tag_sentence(self, sentence):
        sent = Sentence(sentence)
        self.ner_tagger.predict(sent)
        self.pos_tagger.predict(sent)
        ner_dict = sent.to_dict(tag_type='ner')
        pos_dict = sent.to_dict(tag_type='pos')
        return ner_dict, pos_dict
    
    def question_parser(self, question):
        triplets = self.get_triplets_from_sentence(question, store=False)
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
    
    args_list = [(tup, key, term, stemmer, stemmed_term) for tup in triplets]
    logger.info("Searching...")
    with multiprocessing.Pool(cpus) as pool:
        results = list(tqdm(pool.imap(wrapped_matcher, args_list,\
                                      chunksize=int(len(args_list)//cpus)), total=len(args_list)))
    
    logger.info("Collecting results...")
    for result in tqdm(results):
        matched_triplets.extend(result)
    
    return matched_triplets

def double_matcher(tup, keys, terms, stemmer, stemmed_terms): # terms, keys >= 2 iterables
    matched_triplets_tmp = []
    target = tup[keys[0]].lower(); target2 = tup[keys[1]].lower()
    stemmed_target = " ".join([stemmer.stem(t) for t in target.split()])
    stemmed_target2 = " ".join([stemmer.stem(t) for t in target2.split()])
    
    if (len(keys) == 3) and (len(terms) == 3):
        target3 = tup[keys[2]].lower()
        stemmed_target3 = " ".join([stemmer.stem(t) for t in target3.split()])
        if (re.search(terms[0], target) is not None) and (re.search(terms[1], target2) is not None) \
            and (re.search(terms[2], target3) is not None):
            matched_triplets_tmp.append(tup)
        elif (re.search(stemmed_terms[0], stemmed_target) is not None) and \
                (re.search(stemmed_terms[1], stemmed_target2) is not None) and \
                (re.search(stemmed_terms[2], stemmed_target3) is not None):
            matched_triplets_tmp.append(tup)
        return matched_triplets_tmp
    else:
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
    
class KB_Bot(Text_KB_Parser):
    def __init__(self, args=None):
        super(KB_Bot, self).__init__()
        if args is None:
            args = load_pickle("./data/args.pkl")
        self.stemmer = PorterStemmer()
        
        if os.path.isfile(args.state_dict):
            self.load_(args)
        else:
            logger.info("Building KB...")
            self.df = load_tagged_data(args)
            
            logger.info("Extracting triplets...")
            cpus = multiprocessing.cpu_count()
            with multiprocessing.Pool(cpus) as pool:
                results = list(tqdm(pool.imap(self.get_triplets_from_sentence, self.df['sents'],\
                                      chunksize=int(len(self.df['sents'])//cpus)), total=len(self.df['sents'])))
            
            logger.info("Collecting results...")
            results_d = []
            _ = [results_d.extend(t) for t in tqdm(results, total=len(results))]
            results = results_d; del results_d
            self.triplets = results; del results
            self.subjects, self.predicates, self.objects = [], [], []
            for s, p, o in tqdm(self.triplets):
                self.subjects.append(s)
                self.predicates.append(p)
                self.objects.append(o)

            self.cleanup_()
            self.save_(args)
            logger.info("Done and saved at %s!" % args.state_dict)
        
        logger.info("\n***Document statistics***")
        logger.info("%d sentences" % len(self.df))
        logger.info("%d characters" % self.df['length'].sum())
        
        logger.info("\n***KB statistics***")
        logger.info("%d subject-predicate-object triplets" % len(self.triplets))
        logger.info("%d subjects" % len(self.subjects))
        logger.info("%d predicates" % len(self.predicates))
        logger.info("%d objects" % len(self.objects))
        logger.info("%d unique entities" % (len(self.subject_entities) +\
                                            len(self.object_entities)))
        
        
    def save_(self, args):
        filename = args.state_dict
        state_dict = {'df': self.df,\
                      'subjects': self.subjects,\
                      'predicates': self.predicates,\
                      'objects': self.objects,\
                      'triplets': self.triplets,\
                      'subject_entities': self.subject_entities,\
                      'object_entities': self.object_entities,\
                      'subject_entities_d': self.subject_entities_d,\
                      'object_entities_d': self.object_entities_d,\
                      }
        save_as_pickle(filename, state_dict)
        logger.info("Saved state dict.")
        return
    
    def load_(self, args):
        state_dict = load_pickle(args.state_dict)
        self.df = state_dict['df']
        self.subjects = state_dict['subjects']
        self.predicates = state_dict['predicates']
        self.objects = state_dict['objects']
        self.triplets = state_dict['triplets']
        self.subject_entities = state_dict['subject_entities']
        self.object_entities = state_dict['object_entities']
        self.subject_entities_d = state_dict['subject_entities_d']
        self.object_entities_d = state_dict['object_entities_d']
        logger.info("Loaded KB from saved file.")
        return
        
    def query(self, term, key='subject'):
        key_dict = {'subject':0, 'predicate':1, 'object':2}
        if isinstance(key, str):
            key_ = key_dict[key]
        elif isinstance(key, list):
            key_ = [key_dict[k] for k in key]
        matched_triplets = parallel_search(term, self.triplets, self.stemmer, key=key_)
        print("%d results found." % len(matched_triplets))
        return matched_triplets
    
    def ask(self, question):
        qns, verb, selected_entity = self.question_parser(str(question))
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
    **To-do-list
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