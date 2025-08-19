import spacy
from typing import Iterable, List, Tuple, Optional
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np


nlp = spacy.blank("en")

def preprocess(text: str,
               lowercase: bool = True,
               remove_stop: bool = True,
               min_token_len: int =2) -> str:
    """
    Preprocessing:
    - tokenize
    - optional lowercasing
    - optional stopword removal
    Returns a space-joined string for TF-IDF
    """
    if not text:
        return ""
    
    doc = nlp.make_doc(text) 
    out = []

    for t in doc:
        if t.is_space:
            continue
        if t.is_alpha:
            tok = t.text.lower() if lowercase else t.text
            if remove_stop and tok in STOP_WORDS:
                continue
            if len(tok) < min_token_len:
                continue
            out.append(tok)
    return " ".join(out)

def preprocess_many(texts: Iterable[str], **kwargs)-> List[str]:
    return [preprocess(t, **kwargs) for t in texts]

def tfidf(docs: List[str],
          max_features: Optional[int]=40000,
          ngram_range: Tuple[int, int] = (1,2),
          min_df: int=4,
          max_df: float = 0.5,
          sublinear_tf: bool = True,
          norm: str = "l2",
          dtype=np.float32) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Fit TF-IDF on preprocessed docs and return (vectorizer, matrix)
    
    """
    vect = TfidfVectorizer(
        max_features = max_features,
        ngram_range = ngram_range,
        min_df = min_df,
        max_df = max_df,
        sublinear_tf = sublinear_tf,
        norm = norm,
        dtype = dtype
    )
    X = vect.fit_transform(docs)
    return vect, X

def reduce_svd(X, n_components):
    """
    Reduce TF-IDF dimensionality for better clustering
    """
    svd = TrucatedSVD(n_components=n_components, random_state=random_state)
    Xr = svd.fit_transform(X)
    return svd, Xr


