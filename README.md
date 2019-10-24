# Knowledge Base Builder
Builds a Knowledge Base from given input corpus, from which applications can be served. Provides analytics insights into text contents.
*Note this repo currently in development. (see [To do list](#to-do-list)) 

---

## Contents
**Tasks**:  
1. [Classification](#1-classification)
2. [Automatic Speech Recognition](#2-automatic-speech-recognition)
3. [Text Summarization](#3-text-summarization)
4. [Machine Translation](#4-machine-translation)
5. [Natural Language Generation](#5-natural-language-generation)
6. [Punctuation Restoration](#6-punctuation-restoration)  
7. [Named Entity Recognition](#7-named-entity-recognition)
  
[Benchmark Results](#benchmark-results)  
[References](#references)

---

## Pre-requisites
torch==1.2.0 ; spacy==2.1.8 ; torchtext==0.4.0 ; seqeval==0.0.12 ; pytorch-nlp==0.4.1  
For mixed precision training (-fp16=1), apex must be installed: [apex==0.1](https://github.com/NVIDIA/apex)  
For chinese support in Translation: jieba==0.39  
For ASR: librosa==0.7.0 ; soundfile==0.10.2  
For more details, see requirements.txt

** Pre-trained models (XLNet, BERT, GPT-2) are courtesy of huggingface (https://github.com/huggingface/pytorch-transformers)

## Package Installation
```bash
git clone https://github.com/plkmo/NLP_Toolkit.git
cd NLP_Toolkit
pip install .

# to uninstall if required to re-install after updates,
# since this repo is still currently in active development
pip uninstall nlptoolkit 
```
Alternatively, you can just use it as a non-packaged repo after git clone.

---

# To do list
In order of priority:
- [ ] Include package usage info for ~~classification~~, ASR, summarization, ~~translation~~, ~~generation~~, ~~punctuation_restoration~~, ~~NER~~, POS
- [ ] Include benchmark results for  ~~classification~~, ASR, summarization, translation, generation, ~~punctuation_restoration~~, ~~NER~~, POS
- [ ] Include pre-trained models + demo based on benchmark datasets for ~~classification~~, ASR, summarization, translation, ~~generation~~, punctuation_restoration, NER, POS
- [ ] Include more models for punctuation restoration, translation, NER

