import itertools
from .base import Probe


class ConstVar:
    def __init__(self, axiom, epsilon=0):
        self.axiom = axiom
        self.epsilon = epsilon

    def score(self, query, doc_id, rel):
        raise NotImplementedError

    def is_const(self, a, b):
        return abs(a - b) <= self.epsilon

    def is_var(self, a, b):
        return a - self.epsilon > b

    def sort_key(self, val):
        return val


class Len(ConstVar):
    def __init__(self, axiom, epsilon=0):
        super().__init__(axiom, epsilon)
        self.index = axiom.dataset.docs_ptindex()

    def score(self, query, doc_id, rel):
        return len(self.index.doc(doc_id))


class Tf(ConstVar):
    def __init__(self, axiom, epsilon=0):
        super().__init__(axiom, epsilon)
        self.index = axiom.dataset.docs_ptindex()
        self._prev_query = None
        self._prev_qtoks = None

    def score(self, query, doc_id, rel):
        if query['query_text'] != self._prev_query:
            self._prev_query = query['query_text']
            self._prev_qtoks = self.index.tokenize(self._prev_query)
        return self.index.doc(doc_id).tfs(self._prev_qtoks)

    def is_const(self, a, b):
        return all(abs(a.get(t, 0) - b.get(t, 0)) <= self.epsilon for t in a.keys() | b.keys())

    def is_var(self, a, b):
        # all are at least as large, and one is strictly larger
        return all(a.get(t, 0) - self.epsilon >= b.get(t, 0) for t in a.keys() | b.keys()) and \
               any(a.get(t, 0) - self.epsilon > b.get(t, 0) for t in a.keys() | b.keys())

    def sort_key(self, val):
        return sum(val.values())


class SumTf(ConstVar):
    def __init__(self, axiom, epsilon=0):
        super().__init__(axiom, epsilon)
        self.index = axiom.dataset.docs_ptindex()
        self._prev_query = None
        self._prev_qtoks = None

    def score(self, query, doc_id, rel):
        if query['query_text'] != self._prev_query:
            self._prev_query = query['query_text']
            self._prev_qtoks = self.index.tokenize(self._prev_query)
        return sum(self.index.doc(doc_id).tfs(self._prev_qtoks).values())


class Overlap(ConstVar):
    def __init__(self, axiom, epsilon=0):
        super().__init__(axiom, epsilon)
        self.index = axiom.dataset.docs_ptindex()
        self._prev_query = None
        self._prev_qtoks = None

    def score(self, query, doc_id, rel):
        if query['query_text'] != self._prev_query:
            self._prev_query = query['query_text']
            self._prev_qtoks = self.index.tokenize(self._prev_query)
        doc = self.index.doc(doc_id)
        tfs = doc.tfs(self._prev_qtoks)
        if len(doc) > 0:
            return sum(tfs.values()) / len(doc)
        return 0

    def is_const(self, a, b):
        if a == 0. or b == 0.:
            return False # don't do 0s
        return super().is_const(a, b)

    def is_var(self, a, b):
        if a == 0. or b == 0.:
            return False # don't do 0s
        return super().is_var(a, b)


class Rel(ConstVar):
    def score(self, query, doc_id, rel):
        return rel


class ConstVarQrelsProbe(Probe):
    def __init__(self, dataset, const, var, const_epsilon=0, var_epsilon=0, query_field='text', doc_field='text'):
        self.dataset = dataset
        self.constant = const
        self.variable = var
        self.const_epsilon = const_epsilon
        self.var_epsilon = var_epsilon
        self.query_field = query_field
        self.doc_field = doc_field

    def pairs_iter(self):
        qrels = self.dataset.qrels_dict()
        docstore = self.dataset.docs_store()
        const = self.constant(self, self.const_epsilon)
        var = self.variable(self, self.var_epsilon)
        query_field_idx = self.dataset.queries_cls()._fields.index(self.query_field)
        queries_namespace = self.dataset.queries_namespace()
        docs_namespace = self.dataset.docs_namespace()
        for query in self.dataset.queries_iter():
            docs = []
            query = {'query_id': query.query_id, 'query_text': query[query_field_idx]}
            for doc_id, rel in qrels.get(query['query_id'], {}).items():
                docs.append({
                    'id': doc_id,
                    'const': const.score(query, doc_id, rel),
                    'var': var.score(query, doc_id, rel),
                })
            docs = sorted(docs, key=lambda x: const.sort_key(x['const']))
            for i, doc_a in enumerate(docs):
                docs_gt = itertools.takewhile(lambda doc_b: const.is_const(doc_a['const'], doc_b['const']), docs[i+1:])
                docs_lt = itertools.takewhile(lambda doc_b: const.is_const(doc_a['const'], doc_b['const']), docs[i-1::-1])
                for doc_b in itertools.chain(docs_gt, docs_lt):
                    if var.is_var(doc_a['var'], doc_b['var']):
                        yield [
                            dict(**query, doc_id=doc_a['id'], doc_text=docstore.get(doc_a['id'], self.doc_field)),
                            dict(**query, doc_id=doc_b['id'], doc_text=docstore.get(doc_b['id'], self.doc_field)),
                        ]
