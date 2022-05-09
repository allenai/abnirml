import random
import spacy
import ir_datasets
import abnirml
from .base import Probe


class ParaphraseProbe(Probe):
    def __init__(self, dataset='abnirml:mspc', doc_field='text', paraphrase_label=True, query_inferer=None):
        self.dataset = ir_datasets.load(dataset)
        self.doc_field = doc_field
        self.paraphrase_label = paraphrase_label
        self.query_inferer = query_inferer or abnirml.util.CommonNounChunk()

    def pair_symmetry(self):
        return 'symmetric'

    def pairs_iter(self):
        docstore = self.dataset.docs_store()
        for docpair in self.dataset.docpairs_iter():
            if docpair.paraphrase != self.paraphrase_label:
                continue
            doc_a = getattr(docstore.get(docpair.doc_id_a), self.doc_field)
            doc_b = getattr(docstore.get(docpair.doc_id_b), self.doc_field)
            if doc_a != doc_b:
                for query in self.query_inferer.infer_queries(doc_a, doc_b):
                    yield [
                        {'query_text': query, 'doc_text': doc_a},
                        {'query_text': query, 'doc_text': doc_b},
                    ]
