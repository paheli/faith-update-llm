from sentence_transformers import CrossEncoder, SentenceTransformer
from dataset import *
from collections import OrderedDict
from typing import Optional
import numpy as np
import random

def random_rank_evidences(article: ArticleUpdate) -> OrderedDict[EvidenceIndex, Evidence]:
    shuffled = list(article.evidences.keys())
    random.shuffle(shuffled)
    return OrderedDict((k, article.evidences[k]) for k in shuffled) 

def filter_rank_evidences(article: ArticleUpdate, model: Optional[CrossEncoder] = None, threshold=0.5) -> OrderedDict[EvidenceIndex, Evidence]:
    if model is None:
        model = CrossEncoder('vectara/hallucination_evaluation_model')
    e_texts = []
    e_keys = list(article.evidences.keys())
    for e in article.evidences.values():
        text = e.content if isinstance(e.content, str) else e.content.raw_text
        e_texts.append(text)
    
    scores = dict(zip(e_keys, model.predict([[article.normalised_source, t] for t in e_texts])))
    filtered_evidences = sorted([k for k in e_keys if scores[k] <= threshold], key=lambda k: float(scores[k]))
    return OrderedDict((k, article.evidences[k]) for k in filtered_evidences)


def similarity_filter_rank(article: ArticleUpdate, model: Optional[SentenceTransformer] = None, upper_threshold=0.9, lower_threshold=0):
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    e_texts = []
    e_keys = list(article.evidences.keys())
    for e in article.evidences.values():
        text = e.content if isinstance(e.content, str) else e.content.raw_text
        e_texts.append(text)
    source_embeddings = model.encode(article.normalised_source) 
    evidence_embeddings = model.encode(e_texts)
    similarities = model.similarity(source_embeddings, evidence_embeddings)[0]
    k2i = {k: i for i, k in enumerate(e_keys)}
    filtered_evidences = sorted([k for i, k in enumerate(e_keys) if lower_threshold <= similarities[i] <=  upper_threshold], key=lambda k: float(similarities[k2i[k]]))
    return OrderedDict((k, article.evidences[k]) for k in filtered_evidences)
    