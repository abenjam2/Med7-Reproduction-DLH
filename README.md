# Med7-Reproduction-DLH
This is the repository is an official implementation of [Med7: a transferable clinical natural language processing model for electronic health records](https://www.sciencedirect.com/science/article/pii/S0933365721000798).

The data used for this project was obtained from the Medical Information Mart for Intensive Care (MIMIC-III) dataset. More specifically, discharge letters in the Notes Events file within MIMIC-III. Additionally, Track 2 of the 2018 National NLP Clinical Challenges (n2c2) Shared Task on drug related concepts extraction was used to further train the data as well as test the model.

The goal of the original paper is to create a Named-Entity Recognition Model (NER) to extract useful data from Electronic Health Records (EHRs) using Natural Language Processing (NLP). Using spaCY version 3.2.4 and Python 3.7, we were able to run MIMIC-III data through the Med7 vectors model. The model was able to identify Dosage, Drug, Duration, Form, Frequency, Route, and Strength from the unstructured data of EHRs.

# Setup
Before running the Med7 model, the following steps had to be implemented:

1. Install spaCY
pip install -U spacy

2. Install the Med7 Vectors Model
pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl

# Training
The Med7 model comes pre-trained from the initial authors. Additional training can be done using entity ruler (see example below):

```
med7 = spacy.load("en_core_med7_lg")

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
                {"label": "FREQUENCY", "pattern": "every six (6) hours as needed"}]
ruler.add_patterns(train_data)
```

# Evaluation
To produce meaningful results, the output from the model must be separated into correct, incorrect, spurious, incomplete, and missing.

```
potential_missing_count = 0
potential_spurious_count = 0
correct_count = 0
potential_missing_patterns = []
potential_spurious_patterns = []

for i in range(len(test_patterns)):
    pat_to_test = test_patterns[i].lower()
    if pat_to_test not in val_patterns:
        potential_spurious_count+=1
        potential_spurious_patterns.append(pat_to_test)
        
potential_incomplete = 0
potential_extra = 0
extra_patterns = []
incomplete_patterns = []
for i in range(len(potential_spurious_patterns)):
    potential_missing_pattern = potential_spurious_patterns[i]
    for j in range(len(val_patterns)):
        if potential_missing_pattern in val_patterns[j]:
            incomplete_patterns.append(potential_missing_pattern)
        if val_patterns[j] in potential_missing_pattern:
            extra_patterns.append(potential_missing_pattern)
            
unique_incomplete_patterns = []
unique_extra_patterns = []

incomplete = 0
extra = 0

for i in range(len(incomplete_patterns)):
    if incomplete_patterns[i] not in unique_incomplete_patterns:
        unique_incomplete_patterns.append(incomplete_patterns[i])
        incomplete += 1
        
for i in range(len(extra_patterns)):
    if extra_patterns[i] not in unique_extra_patterns:
        unique_extra_patterns.append(extra_patterns[i])
        extra += 1
        
partial = incomplete + extra
spurious_missing = potential_spurious_count - partial
spurious = int(spurious_missing/2)
missing = spurious

correct = 0
correct_patterns = []
correct_labels = []

for i in range(len(test_patterns)):
    pattern_to_test = test_patterns[i]
    label_to_test = test_labels[i]
    if pattern_to_test in val_patterns:
        index = val_patterns.index(pattern_to_test)
        if label_to_test.lower() == val_labels[index].lower():
            correct += 1
            correct_patterns.append(pattern_to_test)
            correct_labels.append(label_to_test)
            
incorrect = len(test_patterns) - (correct + missing + spurious + partial)
```
Then, the different categories can be used to obtain the Strict and Lenient F-1 Scores of an NER Model. We have also calculated the general accuracy of the model as well as looked at the number of classifications per label and compared that with the Gold Standard:
```
possible = correct + incorrect + partial + missing
actual = correct + incorrect + partial + spurious
precision_lenient = (correct + partial)/actual
precision_strict = correct/actual
recall_lenient = (correct + partial)/possible
recall_strict = correct/possible

len_f1 = 2 * (precision_lenient * recall_lenient)/(precision_lenient + recall_lenient)
strict_f1 = 2 * (precision_strict * recall_strict)/(precision_strict + recall_strict)
len_f1
strict_f1

Counter(correct_labels)
Counter(test_labels)
correct/len(test_patterns)
```
# Pre-Trained Models
The Pre-Trained Med7 model can be downloaded [here](https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl). This model was trained on the MIMIC-III dataset describing discharge notes from the ICU.

# Results
Our recreation was able to acheive the below Strict and Lenient F-1 Scores. Our Lenient F-1 Score has exceeded that of the original authors. However, our Strict F-1 Score did not reach that of the original authors.

![image](https://github.com/abenjam2/Med7-Reproduction-DLH/blob/main/Screen%20Shot%202022-05-08%20at%209.19.36%20PM.png)
