import re
import itertools
import random
import string
import spacy
from ..scorers import doctttttquery
from .base import Probe
from abnirml.java import J


class TransformProbe(Probe):
    def __init__(self, dataset, transform, rel_range=None, query_field='text', doc_field='text'):
        self.dataset = dataset
        if rel_range is not None:
            if isinstance(rel_range, (tuple, list)):
                assert len(rel_range) == 2
            else:
                rel_range = (rel_range, rel_range)
        self.rel_range = rel_range
        self.query_field = query_field
        self.doc_field = doc_field
        self.transform = transform

    def pairs_iter(self):
        qrels = self.dataset.qrels_dict()
        docstore = self.dataset.docs_store()
        query_field_idx = self.dataset.queries_cls()._fields.index(self.query_field)
        with self.transform:
            for query in self.dataset.queries_iter():
                query = {'query_id': query.query_id, 'query_text': query[query_field_idx]}
                these_qrels = qrels.get(query['query_id'], {})
                for doc_id, rel in these_qrels.items():
                    if self.rel_range is None or self.rel_range[0] <= rel <= self.rel_range[1]:
                        dtext_a = docstore.get(doc_id, self.doc_field)
                        sample_a = dict(**query, doc_id=doc_id, doc_text=dtext_a)
                        sample_b = self.transform.transform(sample_a, these_qrels, docstore, self.doc_field)
                        if sample_b is not None:
                            yield [sample_a, sample_b]


class CtxtManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Transform(CtxtManager):
    def __init__(self):
        super().__init__()
        self.record = None
        self.qrels = None
        self.docstore = None
        self.doc_field = None

    def transform(self, sample, qrels, docstore, doc_field):
        dtext_a = sample['doc_text']
        self.record = sample
        self.qrels = qrels
        self.docstore = docstore
        self.doc_field = doc_field
        dtext_b = self.transform_text(dtext_a)
        self.record = None
        self.qrels = None
        self.docstore = None
        self.doc_field = None
        if dtext_b and dtext_a != dtext_b:
            return {**sample, 'doc_id': None, 'doc_text': dtext_b}
        return None

    def transform_text(self, text):
        raise NotImplementedError


class SpacyMixin(CtxtManager):
    def __init__(self, **kwargs):
        super().__init__()
        self.spacy_model = kwargs.get('spacy_model', 'en_core_web_sm')
        self.nlp = None

    def __enter__(self):
        super().__enter__()
        self.nlp = spacy.load(self.spacy_model)

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.nlp = None


class RandomMixin(CtxtManager):
    def __init__(self, **kwargs):
        super().__init__()
        self.random_seed = kwargs.get('random_seed', 42)
        self.rng = None

    def __enter__(self):
        super().__enter__()
        self.rng = random.Random(self.random_seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.rng = None



class CaseFold(Transform):
    def transform_text(self, text):
        return text.lower()


class DelPunct(Transform):
    def __init__(self):
        super().__init__()
        self.trans_punct = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    def transform_text(self, text):
        return text.translate(self.trans_punct)


class DelSent(Transform, SpacyMixin, RandomMixin):
    def __init__(self, position='rand', **kwargs):
        super().__init__(**kwargs)
        self.position = position

    def transform_text(self, text):
        sents = list(self.nlp(text).sents)
        if len(sents) > 1: # don't remove if only 1 sentence
            if self.position == 'start':
                sents = sents[1:]
            elif self.position == 'end':
                sents = sents[:-1]
            elif self.position == 'rand':
                pos = self.rng.randrange(len(sents))
                sents = sents[:pos] + sents[pos+1:]
            else:
                raise ValueError()
            return ' '.join(str(s) for s in sents)
        return None


class AddSent(Transform, SpacyMixin, RandomMixin):
    def __init__(self, position='start', rel=0, **kwargs):
        super().__init__(**kwargs)
        self.position = position
        self.rel = rel

    def transform_text(self, text):
        doc_candidates = [did for did, score in self.qrels.items() if score == self.rel]
        if doc_candidates:
            doc_id = self.rng.choice(doc_candidates)
            dtext = self.docstore.get(doc_id, self.doc_field)
            sents = list(self.nlp(dtext).sents)
            sent = self.rng.choice(sents)
            if self.position == 'start':
                text = f'{sent} {text}'
            elif self.position == 'end':
                text = f'{text} {sent}'
            else:
                raise ValueError()
            return text
        return None


class Lemmatize(Transform, SpacyMixin):
    def transform_text(self, text):
        dtext_b = [t.lemma_ if not t.is_stop else t for t in self.nlp(text)]
        return ' '.join(str(s) for s in dtext_b)


class ShufWords(Transform, SpacyMixin, RandomMixin):
    def transform_text(self, text):
        dtoks = [str(t) for t in self.nlp(text)]
        self.rng.shuffle(dtoks)
        return ' '.join(str(s) for s in dtoks)


class ShufWordsKeepSents(Transform, SpacyMixin, RandomMixin):
    def transform_text(self, text):
        self.rng = random.Random(str(self.random_seed) + text)
        dsents = []
        for sent in self.nlp(text).sents:
            sent_toks = [str(s) for s in sent[:-1]]
            self.rng.shuffle(sent_toks)
            sent_toks = sent_toks + [str(sent[-1])]
            dsents.append(' '.join(sent_toks))
        return ' '.join(dsents)


class ShufWordsKeepSentsAndNPs(Transform, SpacyMixin, RandomMixin):
    def transform_text(self, text):
        self.rng = random.Random(str(self.random_seed) + text)
        dsents = []
        parsed_text = self.nlp(text)
        noun_chunks = list(parsed_text.noun_chunks)
        noun_chunk_idxs = set(itertools.chain(*(range(c.start, c.end) for c in noun_chunks)))
        for sent in parsed_text.sents:
            these_noun_chunks = [str(c) for c in noun_chunks if c.start >= sent.start and c.end <= sent.end]
            these_non_noun_chunks = [str(parsed_text[i]) for i in range(sent.start, sent.end - 1) if i not in noun_chunk_idxs]
            sent_toks = these_noun_chunks + these_non_noun_chunks
            self.rng.shuffle(sent_toks)
            sent_toks = sent_toks + [str(sent[-1])]
            dsents.append(' '.join(sent_toks))
        return ' '.join(dsents)


class ShufWordsKeepNPs(Transform, SpacyMixin, RandomMixin):
    def transform_text(self, text):
        self.rng = random.Random(str(self.random_seed) + text)
        parsed_text = self.nlp(text)
        noun_chunks = list(parsed_text.noun_chunks)
        noun_chunk_idxs = set(itertools.chain(*(range(c.start, c.end) for c in noun_chunks)))
        noun_chunks = [str(c) for c in noun_chunks]
        non_noun_chunks = [str(t) for i, t in enumerate(parsed_text) if i not in noun_chunk_idxs]
        toks = noun_chunks + non_noun_chunks
        self.rng.shuffle(toks)
        return ' '.join(toks)


class ShufNPSlots(Transform, SpacyMixin, RandomMixin):
    def transform_text(self, text):
        self.rng = random.Random(str(self.random_seed) + text)
        parsed_text = self.nlp(text)
        noun_chunks = list(parsed_text.noun_chunks)
        noun_chunk_idxs = {}
        for i, np in enumerate(noun_chunks):
            for j in range(np.start, np.end):
                noun_chunk_idxs[j] = i
        chunks = []
        i = 0
        while i < len(parsed_text):
            if i in noun_chunk_idxs:
                chunks.append(noun_chunk_idxs[i])
                i = noun_chunks[noun_chunk_idxs[i]].end
            else:
                chunks.append(str(parsed_text[i]))
                i += 1
        self.rng.shuffle(noun_chunks)
        toks = []
        for chunk in chunks:
            if isinstance(chunk, int):
                toks.append(str(noun_chunks[chunk]))
            else:
                toks.append(chunk)
        return ' '.join(toks)


class ShufPrepositions(Transform, SpacyMixin, RandomMixin):
    def transform_text(self, text):
        self.rng = random.Random(str(self.random_seed) + text)
        parsed_text = self.nlp(text)
        preps = list(t for t in parsed_text if t.pos_ == 'ADP')
        prep_idxs = {}
        for i, prep in enumerate(preps):
            prep_idxs[prep.idx] = i
        chunks = []
        i = 0
        while i < len(parsed_text):
            if i in prep_idxs:
                chunks.append(prep_idxs[i])
            else:
                chunks.append(str(parsed_text[i]))
            i += 1
        self.rng.shuffle(preps)
        toks = []
        for chunk in chunks:
            if isinstance(chunk, int):
                toks.append(str(preps[chunk]))
            else:
                toks.append(chunk)
        return ' '.join(toks)


class SwapNumNPSlots2(Transform, SpacyMixin, RandomMixin):
    def transform_text(self, text):
        self.rng = random.Random(str(self.random_seed) + text)
        parsed_text = self.nlp(text)
        num_swaps = len(list(parsed_text.noun_chunks))
        toks = [str(t) for t in parsed_text]
        new_toks = [str(t) for t in parsed_text]
        positions = self.rng.sample(range(len(toks)), k=num_swaps)
        shuf_positions = list(positions)
        self.rng.shuffle(shuf_positions)
        for old, new in zip(positions, shuf_positions):
            new_toks[new] = toks[old]
        return ' '.join(new_toks)


class ReverseNPSlots(Transform, SpacyMixin):
    def transform_text(self, text):
        parsed_text = self.nlp(text)
        noun_chunks = list(parsed_text.noun_chunks)
        noun_chunk_idxs = {}
        for i, np in enumerate(noun_chunks):
            for j in range(np.start, np.end):
                noun_chunk_idxs[j] = i
        chunks = []
        i = 0
        while i < len(parsed_text):
            if i in noun_chunk_idxs:
                chunks.append(noun_chunk_idxs[i])
                i = noun_chunks[noun_chunk_idxs[i]].end
            else:
                chunks.append(str(parsed_text[i]))
                i += 1
        noun_chunks = list(reversed(noun_chunks))
        toks = []
        for chunk in chunks:
            if isinstance(chunk, int):
                toks.append(str(noun_chunks[chunk]))
            else:
                toks.append(chunk)
        return ' '.join(toks)




class ShufSents(Transform, SpacyMixin, RandomMixin):
    def transform_text(self, text):
        dsents = [str(s) for s in self.nlp(text).sents]
        self.rng.shuffle(dsents)
        dtext_b = ' '.join(str(s) for s in dsents)
        return dtext_b


class ReverseSents(Transform, SpacyMixin):
    def transform_text(self, text):
        dsents = [str(s) for s in self.nlp(text).sents]
        dtext_b = ' '.join(str(s) for s in reversed(dsents))
        return dtext_b


class ReverseWords(Transform, SpacyMixin):
    def transform_text(self, text):
        dtext_b = [str(s) for s in reversed(self.nlp(text))]
        return ' '.join(dtext_b)


class ShufSents(Transform, SpacyMixin, RandomMixin):
    def transform_text(self, text):
        dsents = [str(s) for s in self.nlp(text).sents]
        self.rng.shuffle(dsents)
        dtext_b = ' '.join(str(s) for s in dsents)
        return dtext_b


class Typo(Transform, RandomMixin):
    def __init__(self, no_stops=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._typo_list = None
        self._typo_regex = None
        self._no_stops = no_stops

    def typo_list(self):
        if self._typo_list is None:
            self._typo_list = {}
            if self._no_stops:
                J.initialize()
                stopwords = J._autoclass("org.terrier.terms.Stopwords")(None)
            for line in open('etc/wiki_typos.tsv'):
                typo, corrects = line.rstrip().split('\t')
                corrects = corrects.split(', ')
                for correct in corrects:
                    if self._no_stops and stopwords.isStopword(correct.lower()):
                        continue
                    if correct not in self._typo_list:
                        self._typo_list[correct] = []
                    self._typo_list[correct].append(typo)
            self._typo_regex = '|'.join(re.escape(c) for c in self._typo_list)
            self._typo_regex = re.compile(f'\\b({self._typo_regex})\\b')
        return self._typo_list, self._typo_regex

    def transform_text(self, text):
        typos, regex = self.typo_list()
        match = regex.search(text)
        while match:
            typo_candidates = typos[match.group(1)]
            if len(typo_candidates) > 1:
                typo = self.rng.choice(typo_candidates)
            else:
                typo = typo_candidates[0]
            text = text[:match.start()] + typo + text[match.end():]
            match = regex.search(text)
        return text


class DocTTTTTQuery(Transform):
    def __init__(self, count=4):
        super().__init__()
        self.model = None
        self.count = count

    def transform_text(self, text):
        if self.model is None:
            self.model = doctttttquery.DocTTTTTQueryModel(count=self.count)
        exp = self.model.expand_document(text)
        return f'{text} {exp}'


class Query(Transform):
    def transform_text(self, text):
        return self.record['query_text']


class PrependQuery(Transform):
    def transform_text(self, text):
        return self.record['query_text'] + ' ' + text


class RmStops(Transform, SpacyMixin):
    def __init__(self, source='terrier'):
        super().__init__()
        if source == 'terrier':
            self.STOPS = set(x.strip() for x in open('etc/terrier-stops.txt'))
        else:
            raise ValueError(source)

    def transform_text(self, text):
        terms = [str(t) for t in self.nlp(text) if str(t).lower() not in self.STOPS]
        return ' '.join(terms)


class Multi(Transform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __enter__(self):
        for t in self.transforms:
            t.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for t in self.transforms:
            t.__exit__(exc_type, exc_val, exc_tb)

    def transform_text(self, text):
        for t in self.transforms:
            if text:
                text = t.transform_text(text)
        return text
