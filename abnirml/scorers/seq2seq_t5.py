import torch
import torch.nn.functional as F
import transformers
import ir_datasets
from .base import NeuralScorer


_logger = ir_datasets.log.easy()


class Seq2seqT5(NeuralScorer):
    """
    From:
    > Rodrigo Nogueira, Zhiying Jiang, and Jimmy Lin. Document Ranking with a Pretrained
    > Sequence-to-Sequence Model. arxiv 2020.
    """
    def __init__(self, model_base='t5-large', tok_base='t5-large', batch_size=8):
        super().__init__("VanillaT5", batch_size)
        self.model_base = model_base
        self.tok_base = tok_base

    def delta(self):
        return 0.00237513 # 50th percentile delta from msmarco dev

    def build(self):
        if self.model is None:
            with _logger.duration('loading T5 model'):
                self.model = Seq2seqT5Model(self.model_base, self.tok_base)
                if torch.cuda.is_available():
                    self.model.cuda()


class Seq2seqT5Model(torch.nn.Module):
    def __init__(self, model_base, tok_base):
        super().__init__()
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(tok_base)
        if model_base.startswith('/'):
            self.model = transformers.T5ForConditionalGeneration.from_pretrained(None, config=model_base.replace('pytorch_model.bin', 'config.json'), state_dict=torch.load(model_base, map_location=torch.device('cpu') ))
        else:
            self.model = transformers.T5ForConditionalGeneration.from_pretrained(model_base)
        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]

    def forward(self, **inputs):
        enc = self.tokenizer.batch_encode_plus([f'Query: {q} Document: {d} Relevant:' for q, d in zip(inputs['query_text'], inputs['doc_text'])], return_tensors='pt', pad_to_max_length=True)
        enc['decoder_input_ids'] = torch.full(
            (len(inputs['query_text']), 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long
        )
        for field in list(enc):
            enc[field] = enc[field][:, :512] # crop to 512 (max length)
        if torch.cuda.is_available():
            enc = {k: v.cuda() for k, v in enc.items()}
        result, _, _ = self.model(**enc)
        result = result[:, 0, (self.REL, self.NREL)]
        return F.log_softmax(result, dim=1)[:, 0]
