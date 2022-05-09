import json
import itertools
import hashlib
import scipy
import ir_datasets


_logger = ir_datasets.log.easy()


class AxiomEvaluator:
    def __init__(self, scorer, axiom, hash_fn=hashlib.md5, epsilon=1e-6):
        self.scorer = scorer
        self.axiom = axiom
        self.hash_fn = hash_fn
        self.epsilon = epsilon

    def eval(self):
        records, it2 = itertools.tee(self.axiom.axiom_pairs_iter(), 2)
        def flat_records_iter(it):
            for records in it:
                for record in records:
                    yield record['query_text'], record['doc_text']
        score_iter = self.scorer.score_iter(flat_records_iter(it2))
        axiom_scores = []
        axiom_hash = self.hash_fn()
        pos, neg, neu = 0, 0, 0
        for i, record in enumerate(_logger.pbar(records, desc='axiom pairs')):
            assert len(record['samples']) == 2
            # Keep a hash of the query & doc texts used for this axiom. This is used for verifying
            # that other runs of the axiom yield the same records.
            js = [[i, r['query_text'], r['doc_text']] for r in record['samples']]
            axiom_hash.update(json.dumps(js).encode())

            scores = [s for r, s in zip(record, score_iter)]
            axiom_score = scores[0] - scores[1]
            axiom_scores.append(axiom_score)
            delta = self.scorer.delta(record['context'])
            if axiom_score > delta:
                pos += 1
            elif axiom_score < -delta:
                neg += 1
            else:
                neu += 1
        return {
            'pos': pos,
            'neg': neg,
            'neu': neu,
            'score': (pos - neg) / len(axiom_scores) if len(axiom_scores) > 0 else 0,
            'count': len(axiom_scores),
            'mean_diff': sum(axiom_scores) / len(axiom_scores) if len(axiom_scores) > 0 else 0,
            'median_diff': sorted(axiom_scores)[len(axiom_scores) // 2] if len(axiom_scores) > 0 else 0,
            'p_val': scipy.stats.ttest_1samp(axiom_scores, 0)[1] if len(axiom_scores) > 0 else 0,
            'hash': axiom_hash.hexdigest(),
        }
