import random
import spacy
import ir_datasets
import abnirml
from .base import Probe


class BiasProbe(Probe):
    def __init__(self, dataset='abnirml:nbias', doc_field='text', query_inferer=None):
        self.dataset = ir_datasets.load(dataset)
        self.doc_field = doc_field
        self.query_inferer = query_inferer or abnirml.util.CommonNounChunk()

    def pairs_iter(self):
        for doc in self.dataset.docs_iter():
            doc_a = doc.neutral_text
            doc_b = doc.biased_text
            if doc_a != doc_b:
                for query in self.query_inferer.infer_queries(doc_a, doc_b):
                    yield [
                        {'query_text': query, 'doc_text': doc_a},
                        {'query_text': query, 'doc_text': doc_b},
                    ]
