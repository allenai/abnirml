import random
import spacy
import ir_datasets
import abnirml
from .base import Probe


class JflegProbe(Probe):
    def __init__(self, source='abnirml:jfleg', query_inferer=None):
        self.dataset = ir_datasets.load(source)
        self.query_inferer = query_inferer or abnirml.util.CommonNounChunk()

    def pairs_iter(self):
        for doc in self.dataset.docs_iter():
            text_a = doc.nonfluent
            for text_b in doc.fluents:
                for query in self.query_inferer.infer_queries(text_a, text_b):
                    yield [
                        {'query_text': query, 'doc_text': text_a},
                        {'query_text': query, 'doc_text': text_b},
                    ]
