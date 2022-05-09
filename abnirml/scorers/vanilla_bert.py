import os
import math
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertForNextSentencePrediction
from pytorch_transformers.modeling_bert import BertForPreTraining, BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPreTrainingHeads
from .base import NeuralScorer


class VanillaBERT(NeuralScorer):
    def __init__(self, model_base='bert-base-uncased', weight_path=None, batch_size=8, outputs=2, delta=0.):
        super().__init__("VanillaBERT", batch_size)
        self.model_base = model_base
        self.weight_path = weight_path
        self.outputs = outputs
        self._delta = delta

    def delta(self):
        return self._delta

    def build(self):
        if self.model is None:
            self.model = VanillaBERTModel(self.model_base, self.outputs)
            if self.weight_path is not None:
                self.model.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu')))
                if torch.cuda.is_available():
                    self.model.cuda()


class UntunedBERT(NeuralScorer):
    def build(self):
        if self.model is None:
            self.model = UntunedBERTModel()
            self.model.cuda()


class UntunedBERTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

    def forward(self, query_text, doc_text):
        enc = self.tokenizer.batch_encode_plus(list(zip(query_text, doc_text)), return_tensors='pt', padding=True, truncation=True)
        enc = {k: v[:, :500] for k, v in enc.items()}
        if torch.cuda.is_available():
            enc = {k: v.cuda() for k, v in enc.items()}
        result, = self.model(**enc)
        return F.log_softmax(result, dim=1)[:, 0]


class VanillaBERTModel(torch.nn.Module):
    def __init__(self, bert_base, outputs=2):
        super().__init__()
        self.encoder = JointBertEncoder(bert_base)
        self.ranker = torch.nn.Linear(1024 if 'large' in bert_base else 768, outputs)

    def forward(self, **inputs):
        pooled_output = self.encoder.enc_query_doc(**inputs)['cls']
        result = self.ranker(pooled_output)
        return result[:, 0]


class JointBertEncoder(torch.nn.Module):

    def __init__(self, bert_base):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_base)
        self.bert = CustomBertModelWrapper.from_pretrained(bert_base)
        self.CLS = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.SEP = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.bert.set_trainable(True)

    def enc_query_doc(self, **inputs):
        enc = self.tokenizer.batch_encode_plus(list(zip(inputs['query_text'], inputs['doc_text'])), return_tensors='pt', padding=True, truncation=True)
        enc = {k: v[:, :500] for k, v in enc.items()}
        if torch.cuda.is_available():
            enc = {k: v.cuda() for k, v in enc.items()}
        result = self.bert(**enc)
        return {
            'cls': result[-1][:, 0]
        }


def subbatch(toks, maxlen):
    _, DLEN = toks.shape[:2]
    SUBBATCH = math.ceil(DLEN / maxlen)
    S = math.ceil(DLEN / SUBBATCH) if SUBBATCH > 0 else 0 # minimize the size given the number of subbatch
    stack = []
    if SUBBATCH == 1:
        return toks, SUBBATCH
    else:
        for s in range(SUBBATCH):
            stack.append(toks[:, s*S:(s+1)*S])
            if stack[-1].shape[1] != S:
                nulls = torch.zeros_like(toks[:, :S - stack[-1].shape[1]])
                stack[-1] = torch.cat([stack[-1], nulls], dim=1)
        try:
            return torch.cat(stack, dim=0), SUBBATCH
        except:
            import pdb; pdb.set_trace()
            pass


def un_subbatch(embed, toks, maxlen):
    BATCH, DLEN = toks.shape[:2]
    SUBBATCH = math.ceil(DLEN / maxlen)
    if SUBBATCH == 1:
        return embed
    else:
        embed_stack = []
        for b in range(SUBBATCH):
            embed_stack.append(embed[b*BATCH:(b+1)*BATCH])
        embed = torch.cat(embed_stack, dim=1)
        embed = embed[:, :DLEN]
        return embed

def lens2mask(lens, size):
    mask = []
    for l in lens.cpu():
        l = l.item()
        mask.append(([1] * l) + ([0] * (size - l)))
    return torch.tensor(mask, device=lens.device).long()


class CustomBertModelWrapper(BertForPreTraining):
    def __init__(self, config, depth=None):
        config.output_hidden_states = True
        super().__init__(config)
        self.bert = CustomBertModel(config, depth) # replace with custom model

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.bert(input_ids, token_type_ids, attention_mask)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        result = super().from_pretrained(*args, **kwargs)
        if result.bert.depth is not None:
            # limit the depth by cutting out layers it doesn't need to calculate
            result.bert.encoder.layer = result.bert.encoder.layer[:result.bert.depth]
        else:
            result.depth = len(result.bert.encoder.layer)
        return result

    def set_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = trainable


class CustomBertModel(BertPreTrainedModel):
    """
    Based on pytorch_pretrained_bert.BertModel, but with some extra goodies:
     - depth: number of layers to run in BERT, where 0 is the raw embeddings, and -1 is all
              available layers
    """
    def __init__(self, config, depth=None):
        super(CustomBertModel, self).__init__(config)
        self.depth = depth
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.cls = BertPreTrainingHeads(config)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        Based on pytorch_pretrained_bert.BertModel
        """
        embedding_output = self.embeddings(input_ids, token_type_ids)
        if self.depth == 0:
            return [embedding_output]

        return self.forward_from_layer(embedding_output, attention_mask)

    def forward_from_layer(self, embedding_output, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers

        _, encoded_layers = self.encoder(embedding_output, extended_attention_mask, head_mask)
        return list(encoded_layers)
