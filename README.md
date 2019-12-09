# Knowledge Base Builder
Builds a Knowledge Base from a given input corpus, from which different applications can be served.  
Also provides flexible text query capabilities (subject, predicate, object, Q & A) as well as analytics insights into the text contents.

---

## Contents
Python API
1. [Initialize](#Initialize)
2. [Get Subject-Predicate-Object triplets](#Get-Subject-Predicate-Object-triplets)
3. [Get Subjects, Predicates, Objects](#Get-Subjects,-Predicates,-Objects)
4. [Get entities](#Get-entities)
5. [Search for specific terms in subject/predicate/object](#Search-for-specific-terms-in-subject/predicate/object)
6. [Question & Answer](#Question-&-Answer)

Django Web App
7. [Django Web Application](#Django-Web-Application)

## Pre-requisites
networkx==2.3 ; spacy==2.1.8 ; nltk==3.4.4  
For more details, see requirements.txt

## Package Installation
```bash
git clone https://github.com/plkmo/Knowledge-Base-Builder.git
cd Knowledge-Base-Builder
pip install .

# to uninstall if required to re-install after updates,
pip uninstall kbuilder 
```
Alternatively, you can just use it as a non-packaged repo after git clone.

## Setup Django web app
Requires Django installation, Django channels (https://channels.readthedocs.io/en/latest/introduction.html), Docker
```bash
cd kbuilder_django
python manage.py migrate
python manage.py makemigrations KB
python manage.py sqlmigrate KB 0001
docker run -p 6379:6379 -d redis:2.8
python manage.py runserver
```
Open browser at http://127.0.0.1:8000/KB/
---

## Usage
### Initialize
Input text file to be processed.
```python
from kbuilder.src.utils import Config
from kbuilder.src.KB_funcs import KB_Bot

config = Config()
config.text_file = './data/text.txt' # input text file
config.state_dict = './data/KB_Bot_state_dict.pkl' # KB save file
bot = KB_Bot()
```

### Get Subject-Predicate-Object triplets
```python
bot.triplets[:5] # first 5 triplets
```
Output:
```bash
[('he',
  'turned',
  'At the end of his diatribe against the governor and colonialism to me the Member for Tanjong Pagar who has plagued me so consistently and so vociferously in the past but is virtually the leader of the opposition in the eyes of the public'),
 ('They',
  'tolerate',
  'any challenge to their hold on their Malay political base'),
 ('I',
  'wrote',
  'a note authorising Keng Swee to discuss with Razak , Ismail and such other federal ministers of comparable authority concerned in these matters in the central government any proposal for any constitutional rearrange\xad ments of Malaysia'),
 ('it',
  'establish',
  'The day Singapore gets independence diplomatic relations with the countries we oppose'),
 ('The Tunku',
  'was',
  'was quietly talking to Keng Swee about Singapore hiving off .')]
```

### Get Subjects, Predicates, Objects
```python
bot.subjects[:5] # first 5 subjects
```
Output:
```bash
['the colonies',
 'The Ivory Coast',
 'One man who almost understood what had happened and why it did',
 'the Soviet Union',
 'The verdict of the people']
```

```python
bot.predicates[:5] # first 5 predicates
```
Output:
```bash
['keeping', 'patted', 'twisted', 'recognise', 'remember']
```

```python
bot.objects[:5] # first 5 objects
```
Output:
```bash
['with doubts and hesitations',
 'By about seven o’clock',
 'the new system',
 'him that 517 The Singapore Story discussions had taken place only between our traders and their officials , not with Singapore government officers',
 'that had the French given the Vietnamese their full independence they might not have gone communist']
```

### Get entities
```python
bot.subject_entities[:5] # first 5 entities in subject
```
Output:
```bash
['The Ivory Coast',
 'Khir Johari',
 'Utusan',
 'the Cameron Highlands',
 'the Soviet Union']
```

```python
bot.object_entities[:5] # first 5 entities in object
```
Output:
```bash
['the extra two days',
 'Viscount Head',
 'the Soviet Union',
 '481 m',
 'Kampong Amber']
```

```python
bot.subject_entities_d # entity to subjects mapping
bot.object_entities_d # entity to objects mapping
```

### Search for specific terms in subject/predicate/object
```python
# searches for 'Tunku' in subject, 'was' in predicate, 'Singapore' in object
bot.query(term=['Tunku', 'was', 'Singapore'], \
              key=['subject', 'predicate', 'object'])
```
Output:
```bash
[('The Tunku',
  'was',
  'was quietly talking to Keng Swee about Singapore hiving off .'),
 ('the most devastating blow for the Tunku',
  'was',
  'that the Pap had defeated Umno in all three of its overwhelmingly Malay constituencies , he had specially come down which to Singapore to address on the eve of the election')]
```
### Question & Answer
```python
bot.ask("When was Lee Kuan Yew born?")
```
Output:
```bash
              ***Identified Subject: When
              ***Identified Predicate: born
              ***Identified Object: Lee Kuan Yew

10/25/2019 04:54:47 PM [INFO]: Searching...
10/25/2019 04:54:47 PM [INFO]: Collecting results...
Lee Kuan Yew born in Singapore on 16 September 1923.
```

### Django Web Application
![](https://github.com/plkmo/NLP_Toolkit/blob/master/kbuilder_django/app_screenshot.png) 
