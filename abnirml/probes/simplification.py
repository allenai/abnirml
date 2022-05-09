import random
import spacy
import ir_datasets
import abnirml
from .base import Probe


class SimplificationProbe(Probe):
    def __init__(self, dataset='abnirml:wikiturk', query_inferer=None):
        self.dataset = ir_datasets.load(dataset)
        self.query_inferer = query_inferer or abnirml.util.CommonNounChunk()

    def pairs_iter(self):
        for doc in self.dataset.docs_iter():
            for simp in doc.simplifications:
                for query in self.query_inferer.infer_queries(doc.source, simp):
                    yield [
                        {'query_text': query, 'doc_text': doc.source},
                        {'query_text': query, 'doc_text': simp},
                    ]
