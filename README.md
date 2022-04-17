# Med7-Reproduction-DLH
This is the repository for the recreation of  Med7: a transferable clinical natural language processing model for electronic health records. [The original paper is linked here](https://www.sciencedirect.com/science/article/pii/S0933365721000798).

The data used for this project was obtained from the Medical Information Mart for Intensive Care (MIMIC-III) dataset. More specifically, discharge letters in the Notes Events file within MIMIC-III.

The goal of the original paper is to create a Named-Entity Recognition Model (NER) to extract useful data from Electronic Health Records (EHRs) using Natural Language Processing (NLP). Using spaCY version 3.2.4 and Python 3.7, we were able to run MIMIC-III data through the Med7 vectors model. The model was able to identify Dosage, Drug, Duration, Form, Frequency, Route, and Strength from the unstructured data of EHRs.

# Setup
Before running the Med7 model, the following steps had to be implemented:

1. Install spaCY
pip install -U spacy

2. Install the Med7 Vectors Model
pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl

# Code
Although the model is pretrained, the below code was used to further train the model, prepare the data for the model, and run the data through the model. The data running through the model should be text data. 

Import spaCY and the other necessary libraries.
```
import spacy
from spacy.scorer import Scorer
from spacy.training import Example
import pandas as pd
```

Prepare the data. Read the data from the file and extract the data that will be run through the model.
```
data = pd.read_csv('NOTEEVENTS.csv',nrows=1000)
discharge_notes = data.iloc[: , -1]
```

Create a dictionary with the labels as the keys. This is where the results from the model will be stored.
```
keyList = ["DOSAGE", "DRUG", "DURATION", "FORM", "FREQUENCY", "ROUTE", "STRENGTH"]
d = {}
for i in keyList:
    d[i] = []
```

Although the model is pretrained, additional training via entity_ruler is done.
```
ruler = med7.add_pipe("entity_ruler")
patterns = [{"label": "DRUG", "pattern": "prednsone"},
            {"label": "DRUG", "pattern": "aspirin"},
            {"label": "DRUG", "pattern": "vitamin b"},
            {"label": "DRUG", "pattern": "flagyl"},
            {"label": "DRUG", "pattern": "lisinopril"},
            {"label": "FORM", "pattern": "solution"},
            {"label": "FORM", "pattern": "tablet"},
            {"label": "FORM", "pattern": "capsule"},
            {"label": "FORM", "pattern": "puff"},
            {"label": "FORM", "pattern": "adhesive patch"},
            {"label": "FORM", "pattern": "disk with device"},
            {"label": "STRENGTH", "pattern": "50mg/2ml"},
            {"label": "STRENGTH", "pattern": "5mg"},
            {"label": "STRENGTH", "pattern": "100 unit/ml"},
            {"label": "STRENGTH", "pattern": "0.05%"},
            {"label": "STRENGTH", "pattern": "25-50mg"},
            {"label": "DURATION", "pattern": "for 3 days"},
            {"label": "DURATION", "pattern": "7 days"},
            {"label": "DURATION", "pattern": "chronic"},
            {"label": "DURATION", "pattern": "x5 days"},
            {"label": "DURATION", "pattern": "for 5 or more days"},
            {"label": "ROUTE", "pattern": "PO"}, 
            {"label": "ROUTE", "pattern": "iv"},
            {"label": "ROUTE", "pattern": "gtt"},
            {"label": "ROUTE", "pattern": "nasal canula"},
            {"label": "ROUTE", "pattern": "injection"},
            {"label": "DOSAGE", "pattern": "1-2"},
            {"label": "DOSAGE", "pattern": "sliding scale"},
            {"label": "DOSAGE", "pattern": "taper"},
            {"label": "DOSAGE", "pattern": "bolus"},
            {"label": "DOSAGE", "pattern": "thirty (30) ml"},
            {"label": "FREQUENCY", "pattern": "once a day"},
            {"label": "FREQUENCY", "pattern": "b.i.d."},
            {"label": "FREQUENCY", "pattern": "prn"},
            {"label": "FREQUENCY", "pattern": "q6h"},
            {"label": "FREQUENCY", "pattern": "hs"},
            {"label": "FREQUENCY", "pattern": "every six (6) hours as needed"},]
ruler.add_patterns(patterns)
```

Load the model, run each entry in the data through the model, and store the results in the dictionary created above. Each result will be placed as a value with its corresponding key (label). A time function is used to view basic computational requirements.
```
%%time
med7 = spacy.load("en_core_med7_lg")
for entry in discharge_notes:
    doc = med7(str(entry))
    [d[ent.label_].append(ent.text) for ent in doc.ents]
d
```
