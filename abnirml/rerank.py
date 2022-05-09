import argparse
import itertools
import ir_datasets
import abnirml


_logger = ir_datasets.log.easy()


def flush(fout, qid, docs):
    scored_docs = [(score, did) for did, score in docs.items()]
    for i, (score, did) in enumerate(sorted(scored_docs, reverse=True)):
        fout.write(f'{qid} 0 {did} {i+1} {score} run\n')
    fout.flush()


def iter_scores(scoreddocs, scorer, query_lookup, doc_store):
    it1, it2 = itertools.tee(scoreddocs)
    def score_input_iter():
        for record in it1:
            yield query_lookup[record.query_id], doc_store.get(record.doc_id, 'text')
    score_iter = scorer.score_iter(score_input_iter())
    for record, score in zip(it2, score_iter):
        yield record.query_id, record.doc_id, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('scorer')
    parser.add_argument('output')
    args = parser.parse_args()
    dataset = ir_datasets.load(args.dataset)
    scorer = abnirml.SCORERS[args.scorer] # remove cacher
    query_lookup = {q.query_id: q.text for q in dataset.queries_iter()}
    doc_store = dataset.docs_store()
    last_qid = None
    doc_scores = None
    with open(args.output, 'wt') as fout:
        results = {}
        for qid, did, score in iter_scores(dataset.scoreddocs_iter(), scorer, query_lookup, doc_store):
            if qid not in results:
                results[qid] = {}
            results[qid][did] = score
        for qid, doc_scores in results.items():
            flush(fout, qid, doc_scores)


if __name__ == '__main__':
    main()
