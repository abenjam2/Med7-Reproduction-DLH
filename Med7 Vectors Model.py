#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install -U spacy


# In[3]:


pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl


# In[1]:


import spacy
from spacy.scorer import Scorer
from spacy.training import Example
import pandas as pd


# In[2]:


data = pd.read_csv('NOTEEVENTS.csv',nrows=1000)
discharge_notes = data.iloc[: , -1]


# In[3]:


keyList = ["DOSAGE", "DRUG", "DURATION", "FORM", "FREQUENCY", "ROUTE", "STRENGTH"]
  
# initialize dictionary
d = {}
  
# iterating through the elements of list
for i in keyList:
    d[i] = []


# In[4]:


get_ipython().run_cell_magic('time', '', '\nmed7 = spacy.load("en_core_med7_lg")\n\n# create distinct colours for labels\n\nfor entry in discharge_notes:\n    doc = med7(str(entry))\n    [d[ent.label_].append(ent.text) for ent in doc.ents]')


# In[5]:


d


# In[ ]:




