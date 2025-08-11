import spacy
import sklearn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Load English tokenizer, tagger, parse and NER
nlp = spacy.load("en_core_web_sm")

# Process documents
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")
doc = nlp(text)

# 1. Tokenization
for token in doc:
        print(token.text, token.pos_, token.dep_)

# Named Entity Recognition
for ent in doc.ents:
        print(ent.text, ent.label_)