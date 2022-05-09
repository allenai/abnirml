import json
import itertools
import hashlib
import scipy
from collections import Counter
import ir_datasets
import abnirml
import seaborn as sns


_logger = ir_datasets.log.easy()


def _asymmetric_probe_scorer(score_a, score_b):
    return score_a - score_b


def _symmetric_probe_scorer(score_a, score_b):
    return abs(score_a - score_b)


def _prober(scorer, probe):
    if hasattr(scorer, 'transform') and not hasattr(scorer, 'score_iter'):
        scorer = abnirml.PyTerrierScorer(scorer) # allow pyterrier transformers
    probe_scorer = {
        'asymmetric': _asymmetric_probe_scorer,
        'symmetric': _symmetric_probe_scorer,
    }[probe.pair_symmetry()]
    records, it2 = itertools.tee(probe.pairs_iter(), 2)
    def flat_records_iter(it):
        for record in it:
            assert len(record) == 2
            for rec in record:
                yield rec['query_text'], rec['doc_text']
    score_iter = scorer.score_iter(flat_records_iter(it2))
    probe_scores = []
    probe_hash = hashlib.md5()
    for i, record in enumerate(_logger.pbar(records, desc='axiom pairs')):
        # Keep a hash of the query & doc texts used for this axiom. This is used for verifying
        # that other runs of the axiom yield the exact same records.
        probe_hash.update(json.dumps([[i, r['query_text'], r['doc_text']] for r in record]).encode())

        scores = [s for r, s in zip(record, score_iter)]
        probe_score = probe_scorer(scores[0], scores[1])
        probe_scores.append(probe_score)
    return {
        'scores': probe_scores,
        'hash': probe_hash.hexdigest(),
    }



def ProbeExperiment(scorer, probe, delta=0.):
    probe_info = _prober(scorer, probe)
    probe_scores = probe_info['scores']
    pos = sum(1 for s in probe_scores if s > delta)
    neg = sum(1 for s in probe_scores if s < -delta)
    neu = sum(1 for s in probe_scores if -delta <= s <= delta)
    return {
        'pos': pos,
        'neg': neg,
        'neu': neu,
        'score': (pos - neg) / len(probe_scores) if len(probe_scores) > 0 else 0,
        'count': len(probe_scores),
        'mean_diff': sum(probe_scores) / len(probe_scores) if len(probe_scores) > 0 else 0,
        'median_diff': sorted(probe_scores)[len(probe_scores) // 2] if len(probe_scores) > 0 else 0,
        'p_val': scipy.stats.ttest_1samp(probe_scores, 0)[1] if len(probe_scores) > 0 else 0,
        'hash': probe_info['hash'],
    }


def ProbeDist(scorer, probes):
    all_scores = []
    for probe in probes:
        probe_info = _prober(scorer, probe)
        sns.kdeplot(probe_info['scores'])
        all_scores.append(probe_info['scores'])
    return scipy.stats.mannwhitneyu(all_scores[0], all_scores[1])


def rerank(scorer, dataset):
    docs = dataset.docs_store()
    queries = {q.query_id: q for q in dataset.queries_iter()}
    records = []
    inputs = []
    for s in dataset.scoreddocs_iter():
        records.append(s)
        inputs.append((queries[s.query_id].text, docs.get(s.doc_id).text))
    score_iter = scorer.score_iter(inputs)
    results = {}
    for record, score in zip(_logger.pbar(records, desc='calculating scores'), score_iter):
        if record.query_id not in results:
            results[record.query_id] = {}
        results[record.query_id][record.doc_id] = score
    return results


def topk_diffs(scorer, dataset, k=10):
    diffs = []
    for scoreddocs in rerank(scorer, dataset).values():
        scoreddocs = Counter(scoreddocs)
        topk = list(scoreddocs.most_common(k))
        for (_, s0), (_, s1) in zip(topk, topk[1:]):
            diffs.append(s0 - s1)
    return diffs


def CutoffCompare(probe, scorers_cutoffs):
    mn, mx = 1, -1
    for n, scorer, cutoffs in scorers_cutoffs:
        probe_info = abnirml.eval.probe_experiment._prober(scorer, probe)
        x, y = [], []
        for i in range(100):
            if i == 99:
                cutoff = cutoffs[-1]
            if i == 0:
                cutoff = 0
            else:
                cutoff = cutoffs[math.floor(len(cutoffs) * i / 100)]
            pos = sum(1 for x in probe_info['scores'] if x > cutoff)
            neg = sum(1 for x in probe_info['scores'] if x < -cutoff)
            x.append(100 - i)
            y.append((pos - neg) / len(probe_info['scores']))
        plt.plot(x, y, label=n)
        mn = min(mn, *y)
        mx = max(mx, *y)
    mx = 0.1 * math.ceil(mx/0.1)
    mn = 0.1 * math.floor(mn/0.1)
    plt.ylim(mn, mx)
    plt.axhline(0, c='#444444', ls=':')
    plt.axvline(50, c='#444444', ls=':')
    plt.xlabel('Percentile')
    plt.ylabel('Score')
    plt.legend()
