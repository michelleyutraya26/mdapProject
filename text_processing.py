import spacy
from spacy.lang.en.stop_words import STOP_WORDS


nlp = spacy.blank("en")

def preprocess(text: str):
    doc = nlp.make_doc(text)  
    tokens = [t.text.lower() for t in doc if t.is_alpha and t.text.lower() not in STOP_WORDS]
    return tokens 

