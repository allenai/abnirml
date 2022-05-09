import torch
import pandas as pd
import pyterrier as pt


class Scorer:
    def __init__(self, name):
        self.name = name

    def delta(self):
        return 0.

    def score_iter(self, it):
        raise NotImplementedError


class NeuralScorer(Scorer):
    def __init__(self, name, batch_size=8):
        super().__init__(name)
        self.batch_size = batch_size
        self.model = None

    def build(self):
        pass

    def score(self, query, doc):
        return self.batch_score([query], [doc])[0]

    def batch_score(self, queries, texts):
        self.build()
        with torch.no_grad():
            return self.model(query_text=queries, doc_text=texts).tolist()

    def score_iter(self, it):
        batch_size = self.batch_size
        q_batch, d_batch = [], []
        for query, doc in it:
            q_batch.append(query)
            d_batch.append(doc)
            if len(q_batch) >= batch_size:
                scores = self.batch_score(q_batch, d_batch)
                yield from scores
                q_batch, d_batch = [], []
        if q_batch:
            scores = self.batch_score(q_batch, d_batch)
            yield from scores


class PyTerrierScorer(Scorer):
    def __init__(self, name, transformerfn=None, delta=0., batch_size=None):
        if transformerfn is None:
            name, transformerfn = "", name
        super().__init__(name)
        self.transformerfn = transformerfn
        self.transformer = None
        if batch_size is None and hasattr(transformerfn, 'batch_size'):
            batch_size = transformerfn.batch_size
        else:
            batch_size = 64
        self.batch_size = batch_size
        self._delta = delta

    def delta(self):
        return self._delta

    def score_iter(self, it):
        if not pt.started():
            pt.init()
        batch_size = self.batch_size
        q_batch, d_batch = [], []
        for query, doc in it:
            q_batch.append(query)
            d_batch.append(doc)
            if len(q_batch) >= batch_size:
                scores = self.batch_score(q_batch, d_batch)
                yield from scores
                q_batch, d_batch = [], []
        if q_batch:
            scores = self.batch_score(q_batch, d_batch)
            yield from scores

    def batch_score(self, queries, texts):
        ids = [str(i) for i in range(len(queries))]
        df = pd.DataFrame({'qid': ids, 'docno': ids, 'query': queries, 'text': texts})
        if self.transformer is None:
            if hasattr(self.transformerfn, 'transform'):
                self.transformer = self.transformerfn
            else:
                self.transformer = self.transformerfn()
        return self.transformer(df)['score'].tolist()
