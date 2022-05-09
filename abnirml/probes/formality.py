import spacy
import ir_datasets
from .base import Probe


class GyafcProbe(Probe):
    def __init__(self, spacy_model='en_core_web_sm', genre_filter=None, yahoo_l6_dataset=None, gyafc_dataset=None):
        self.spacy_model = spacy_model
        self.genre_filter = genre_filter
        self.yahoo_l6_dataset = yahoo_l6_dataset or ir_datasets.load('abnirml:yahoo-l6')
        self.gyafc_dataset = gyafc_dataset or ir_datasets.load('abnirml:gyafc')

    def pairs_iter(self):
        l6_docstore = self.yahoo_l6_dataset.docs_store()
        nlp = spacy.load(self.spacy_model, disable=["parser"])
        for gyafc_doc in self.gyafc_dataset.docs_iter():
            if gyafc_doc.mapped_l6_id == '_':
                # _ indicates not match was found
                continue
            if self.genre_filter is not None and gyafc_doc.genre != self.genre_filter:
                # ignore this genre
                continue
            src_question = l6_docstore.get(gyafc_doc.mapped_l6_id, 'subject')
            src_question_lemmas = {t.lemma_.lower() for t in nlp(src_question) if not t.is_stop and not t.is_punct and not t.like_num and str(t).strip() != ''}
            formal_lemmas = {t.lemma_ for t in nlp(gyafc_doc.formal)}
            informal_lemmas = {t.lemma_ for t in nlp(gyafc_doc.informal)}
            if not src_question_lemmas & formal_lemmas & informal_lemmas:
                # no ovelapping terms among the query and 2 documents
                continue
            yield [
                {'query_text': src_question, 'doc_text': gyafc_doc.formal},
                {'query_text': src_question, 'doc_text': gyafc_doc.informal},
            ]
