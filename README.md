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
