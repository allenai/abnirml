from .base import Probe


class NLAugProbe(Probe):
    def __init__(self, dataset, generator, rel_range=None, query_field='text', doc_field='text'):
        self.dataset = dataset
        self.generator = generator
        if rel_range is not None:
            if isinstance(rel_range, (tuple, list)):
                assert len(rel_range) == 2
            else:
                rel_range = (rel_range, rel_range)
        self.rel_range = rel_range
        self.query_field = query_field
        self.doc_field = doc_field

    def pairs_iter(self):
        qrels = self.dataset.qrels_dict()
        docstore = self.dataset.docs_store()
        query_field_idx = self.dataset.queries_cls()._fields.index(self.query_field)
        for query in self.dataset.queries_iter():
            query = {'query_id': query.query_id, 'query_text': query[query_field_idx]}
            these_qrels = qrels.get(query['query_id'], {})
            for doc_id, rel in these_qrels.items():
                if self.rel_range is None or self.rel_range[0] <= rel <= self.rel_range[1]:
                    dtext_a = docstore.get(doc_id, self.doc_field)
                    sample_a = dict(**query, doc_id=doc_id, doc_text=dtext_a)
                    for generated_text in generator.generate(dtext_a):
                        if text != generated_text:
                            sample_b = dict(**query, doc_text=generated_text)
                            yield [sample_a, sample_b]
