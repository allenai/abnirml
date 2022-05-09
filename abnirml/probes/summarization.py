import ir_datasets
import os
import hashlib
from glob import glob
import spacy
from .base import Probe



class XSumProbe(Probe):
    def __init__(self, spacy_model='en_core_web_sm', dataset='abnirml:xsum'):
        super().__init__()
        self.spacy_model = spacy_model
        self.dataset = ir_datasets.load(dataset)

    def pairs_iter(self):
        nlp = spacy.load(self.spacy_model, disable=["parser"])
        for doc in self.dataset.docs_iter():
            m = hashlib.sha256(f'{doc.doc_id}.summary'.encode())
            if m.digest()[0] % 10 != 0:
                continue # Sample 10%
            title_lemmas = {t.lemma_.lower() for t in nlp(doc.title) if not t.is_stop and not t.is_punct and not t.like_num and str(t).strip() != ''}
            summary_lemmas = {t.lemma_ for t in nlp(doc.first_sentence)}
            content_lemmas = {t.lemma_ for t in nlp(doc.rest_body)}
            if not (title_lemmas & summary_lemmas & content_lemmas):
                # no ovelapping terms among the query and 2 documents
                continue
            if doc.title and doc.rest_body and doc.first_sentence:
                yield [
                    {'query_text': doc.title, 'doc_text': doc.first_sentence},
                    {'query_text': doc.title, 'doc_text': doc.rest_body},
                ]



class CnnDmProbe(Probe):
    def __init__(self, spacy_model='en_core_web_sm', source='both', dataset='abnirml:cnn_dailymail'):
        super().__init__()
        assert source in ('both', 'cnn', 'dm')
        self.spacy_model = spacy_model
        self.source = source
        self.dataset = ir_datasets.load(dataset)

    def pairs_iter(self):
        nlp = spacy.load(self.spacy_model, disable=["parser"])
        for doc in self.dataset.docs_iter():
            if self.source == 'cnn' and not doc.doc_id.startswith('cnn:'):
                continue
            if self.source == 'dm' and not doc.doc_id.startswith('dailymail:'):
                continue
            m = hashlib.sha256(doc.doc_id.split(':')[1].encode())
            if m.digest()[0] % 10 != 0:
                continue # Sample 10%
            title_lemmas = {t.lemma_.lower() for t in nlp(doc.title) if not t.is_stop and not t.is_punct and not t.like_num and str(t).strip() != ''}
            summary_lemmas = {t.lemma_ for t in nlp(doc.summary)}
            content_lemmas = {t.lemma_ for t in nlp(doc.body)}
            if not (title_lemmas & summary_lemmas & content_lemmas):
                # no ovelapping terms among the query and 2 documents
                continue
            yield [
                {'query_text': doc.title, 'doc_text': doc.summary},
                {'query_text': doc.title, 'doc_text': doc.body},
            ]
