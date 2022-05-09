import hashlib
import gzip
import torch
import transformers
import ir_datasets
from .base import Scorer
from .cached import SimpleFsCache


_logger = ir_datasets.log.easy()


class DocTTTTTQuery(Scorer):
    def __init__(self, scorer, count=4, delta=0):
        super().__init__(None)
        self.scorer = scorer
        self.count = count
        self.model = None
        self.cache = None
        self._delta = delta

    def delta(self):
        return self._delta

    def score_iter(self, it):
        self.build()
        def iter_exp_docs():
            for qtext, dtext in it:
                dexp = self.model.expand_document(dtext)
                dtext = f'{dtext} {dexp}'
                yield qtext, dtext
        return self.scorer.score_iter(iter_exp_docs())

    def build(self):
        if self.model is None:
            with _logger.duration('loading T5 model'):
                self.model = DocTTTTTQueryModel(count=self.count)


class DocTTTTTQueryModel:
    def __init__(self, tok_base='t5-base', model_base='castorini/doc2query-t5-base-msmarco', count=4):
        super().__init__()
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(tok_base)
        self.config = transformers.T5Config.from_pretrained('t5-base')
        self.model = transformers.T5ForConditionalGeneration.from_pretrained(model_base, config=self.config)
        if torch.cuda.is_available():
            self.model.cuda()
        self.count = count
        self.cache = SimpleFsCache('cache/doctttttquery.cache', gzip.open)

    def expand_document(self, doc_text):
        key = hashlib.md5(doc_text.encode()).digest()
        if key not in self.cache:
            expansions = []
            expansions.append("")
            doc_text += ' </s>'
            input_ids = self.tokenizer.encode(doc_text, return_tensors='pt')
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            input_ids = input_ids[:, :1024]
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=64,
                do_sample=True,
                top_k=10,
                num_return_sequences=self.count)
            for i in range(self.count):
                exp = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                expansions.append(exp)
            self.cache[key] = ' '.join(expansions)
        return self.cache[key]
