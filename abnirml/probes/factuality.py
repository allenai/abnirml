import re
import random
import spacy
import ir_datasets
import abnirml
from .base import Probe


_logger = ir_datasets.log.easy()


class FactualityProbe(Probe):
    def __init__(self, dataset='dpr-w100/natural-questions/dev', spacy_model='en_core_web_sm', random_seed=42, valid_entities=('PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE'), tokenizer=None):
        self.dataset = ir_datasets.load(dataset)
        self.spacy_model = spacy_model
        self.random_seed = random_seed
        self.valid_entities = valid_entities
        self.tokenizer = tokenizer

    def pairs_iter(self):
        nlp = spacy.load(self.spacy_model, disable=["parser"])
        docs_store = self.dataset.docs_store()
        queries = [q for q in self.dataset.queries_iter() if len(q.answers) == 1]
        query_map = {q.query_id: q for q in queries}
        query_answer_parsed = {q.query_id: nlp(q.answers[0]) for q in _logger.pbar(queries, desc='parsing answers')}
        qids_by_tok_count_ent = {}
        for qid, qtok in query_answer_parsed.items():
            qlen = len(self.tokenizer.tokenize(str(qtok), include_stops=False)) if self.tokenizer else len(qtok)
            ent = [e for e in qtok.ents if e.label_ in self.valid_entities]
            if len(ent) > 0:
                ent = ent[0].label_ if len(ent) > 0 else ''
                key = (qlen, ent)
                qids_by_tok_count_ent.setdefault(key, []).append(qid)
                query_answer_parsed[qid] = key
            else:
                query_answer_parsed[qid] = None
        print({k: len(qids_by_tok_count_ent[k]) for k in qids_by_tok_count_ent})
        qrels = self.dataset.qrels_dict()
        for query in queries:
            if not query_answer_parsed[query.query_id]:
                continue
            these_pos_dids = [d for d, s in qrels[query.query_id].items() if s > 0]
            these_pos_docs = docs_store.get_many(these_pos_dids)
            for did in sorted(these_pos_dids):
                doc = these_pos_docs[did]
                answer_matcher = ' ?'.join([re.escape(a) for a in query.answers[0].split(' ')])
                if not re.search(answer_matcher, doc.text, flags=re.IGNORECASE):
                    _logger.info(f'answer not found in text qid={query.query_id} did={did}')
                    continue
                if any(re.search(r'\b'+re.escape(t)+r'\b', query.text, flags=re.IGNORECASE) for t in query.answers[0].split(' ')):
                    # this filter is a bit aggressive, but needs to be safe to prevent messing up "A or B"-like questions (where the answer is then either A or B).
                    continue
                rng = random.Random(repr((query.query_id, did, self.random_seed)))
                new_qid = rng.choice(qids_by_tok_count_ent[query_answer_parsed[query.query_id]])
                new_answer = query_map[new_qid].answers[0]
                new_text = re.sub(answer_matcher, new_answer, doc.text, flags=re.IGNORECASE)
                if new_text != doc.text: # also handles if this qid is selected
                    yield [
                        {'query_text': query.text, 'doc_text': doc.text},
                        {'query_text': query.text, 'doc_text': new_text},
                    ]
