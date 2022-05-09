import pyterrier
import pandas as pd
from abnirml.java import J
from .base import Scorer


class _TerrerRetriever(Scorer):
    def __init__(self, name, index, batch_size=100):
        super().__init__(name)
        self.index = index
        self.batch_size = batch_size

    def batch_score(self, queries, texts):
        J.initialize()
        ti = self.index
        retr = pyterrier.batchretrieve.TextScorer(
            background_index=ti._index(),
            controls=self._controls(),
            properties=self._properties(),
            num_results=len(queries))
        df = []
        qid_map = {}
        doc_map = {}
        for q, t in zip(queries, texts):
            q = ti.parse_query(q)
            if q not in qid_map:
                qid_map[q] = len(qid_map)
            if t not in doc_map:
                doc_map[t] = len(doc_map)
            df.append((str(qid_map[q]), q, str(doc_map[t]), t))
        df = pd.DataFrame(df, columns=['qid', 'query', 'docno', 'body'], dtype=str)
        run = retr.transform(df)
        result = []
        for tup in df.itertuples():
            r = run[(run.qid == tup.qid) & (run.docno == tup.docno)]['score']
            if len(r) > 0:
                result.append(list(r)[0])
            else:
                result.append(0.)
        return result

    def score_iter(self, it):
        batch_size = self.batch_size
        q_batch, t_batch = [], []
        for query, text in it:
            q_batch.append(query)
            t_batch.append(text)
            if len(q_batch) >= batch_size:
                scores = self.batch_score(q_batch, t_batch)
                yield from scores
                q_batch, t_batch = [], []
        if q_batch:
            scores = self.batch_score(q_batch, t_batch)
            yield from scores

    def _controls(self):
        return {}

    def _properties(self):
        return {}


class TerrierBM25(_TerrerRetriever):
    def __init__(self, index, k1=1.2, b=0.8, batch_size=100, delta=0.0):
        super().__init__("TerrierBM25", index, batch_size)
        self.k1 = k1
        self.b = b
        self._delta = delta

    def delta(self):
        return self._delta

    def _controls(self):
        return {
            'wmodel': 'BM25',
            'c': str(self.b)
        }

    def _properties(self):
        return {
            'bm25.k_1': str(self.k1)
        }
