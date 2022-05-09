import ir_datasets
import random
import spacy


__all__ = ['QueryInferer', 'CommonNounChunk', 'SelectAll', 'RandomSelector']


class QueryInferer:
    def infer_queries(self, text_a, text_b):
        raise NotImplementedError()


class CommonNounChunk(QueryInferer):
    def __init__(self, spacy_model='en_core_web_sm', min_noun_chunk_len=5, selector=None):
        self.nlp = ir_datasets.util.Lazy(lambda: spacy.load(spacy_model))
        self.min_noun_chunk_len = min_noun_chunk_len
        self.selector = selector if selector is not None else RandomSelector()

    def infer_queries(self, text_a, text_b):
        parsed_a = self.nlp()(text_a)
        parsed_b = self.nlp()(text_b)
        noun_chunks_a = set(str(c).lower() for c in parsed_a.noun_chunks if len(str(c)) > self.min_noun_chunk_len)
        noun_chunks_b = set(str(c).lower() for c in parsed_b.noun_chunks if len(str(c)) > self.min_noun_chunk_len)
        candiates = noun_chunks_a & noun_chunks_b
        return self.selector.select(candiates, text_a, text_b)


class SelectAll:
    def select(self, candiates, text_a, text_b):
        return candiates


class RandomSelector:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    def select(self, candiates, text_a, text_b):
        if candiates:
            rng = random.Random(repr((text_a, text_b, self.random_seed)))
            return [rng.choice(sorted(candiates))]
        return []
