# import os
# import torch
# import torch.nn.functional as F
# from transformers import BertTokenizerFast, BertForNextSentencePrediction
# from pytorch_transformers.modeling_bert import BertForPreTraining, BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPreTrainingHeads
# from .base import NeuralScorer


# class EPIC(NeuralScorer):
#     def __init__(self, weight_path=None, batch_size=8):
#         super().__init__("EPIC", batch_size)
#         self.weight_path = weight_path

#     def build(self):
#         if self.model is None:
#             self.model = EpicModel()
#             weight_path = self.weight_path
#             if os.path.exists(weight_path):
#                 self.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
#                 if torch.cuda.is_available():
#                     self.model = self.model.cuda()

#     def delta(self):
#         return 0.294334 # 50th percentile delta from msmarco dev



# class EpicModel(torch.nn.Module):
#     """
#     Implementation of the EPIC model from:
#       > Sean MacAvaney, Franco Maria Nardini, Raffaele Perego, Nicola Tonellotto,
#       > Nazli Goharian, and Ophir Frieder. 2020. Expansion via Prediction of Importance with
#       > Contextualization. In SIGIR.
#     """
#     def __init__(self):
#         super().__init__()
#         self.encoder = SepBertEncoder()
#         self.query_salience = torch.nn.Linear(768, 1)
#         self.doc_salience = torch.nn.Linear(768, 1)
#         self.activ = lambda x: (1. + F.softplus(x)).log()
#         self._nil = torch.nn.Parameter(torch.zeros(1))
#         self.doc_quality = torch.nn.Linear(768, 1)

#     def forward(self, query_text, doc_text):
#         encoded = self.encoder.enc_query_doc(query_text, doc_text)
#         query_vecs = self.query_full_vector(encoded['query'], encoded['query_len'], encoded['query_tok'])
#         doc_vecs, _ = self.doc_full_vector(encoded['doc'], encoded['doc_len'], encoded['doc_cls'])
#         return self.similarity(query_vecs, doc_vecs)

#     def doc_full_vector(self, doc_tok_reps, doc_len, doc_cls):
#         tok_salience = self.doc_salience(doc_tok_reps)
#         tok_salience = self.activ(tok_salience)
#         exp_raw = self.encoder.bert.cls.predictions(doc_tok_reps)
#         mask = lens2mask(doc_len, exp_raw.shape[1])
#         exp = self.activ(exp_raw)
#         exp = exp * tok_salience * mask.unsqueeze(2).float()
#         exp, _ = exp.max(dim=1)
#         qlty = torch.sigmoid(self.doc_quality(doc_cls))
#         exp = qlty * exp
#         qlty = qlty.reshape(doc_cls.shape[0])
#         return exp, qlty

#     def query_full_vector(self, query_tok_reps, query_len, query_tok, dense=True):
#         tok_salience = self._query_salience(query_tok_reps, query_len, query_tok)
#         idx0 = torch.arange(tok_salience.shape[0], device=tok_salience.device).reshape(tok_salience.shape[0], 1).expand(tok_salience.shape[0], tok_salience.shape[1]).flatten()
#         idx1 = query_tok.flatten()
#         idx1[idx1 == -1] = 0

#         s = torch.Size([query_tok.shape[0], self.encoder.lexicon_size()])
#         result = torch.sparse.FloatTensor(torch.stack((idx0, idx1)), tok_salience.flatten(), s)
#         if dense:
#             result = result.to_dense()
#         return result

#     def _query_salience(self, query_tok_reps, query_len, query_tok):
#         inputs = query_tok_reps
#         tok_salience = self.query_salience(inputs)
#         tok_salience = self.activ(tok_salience).squeeze(2)
#         mask = lens2mask(query_len, query_tok.shape[1])
#         tok_salience = tok_salience * mask.float()
#         return tok_salience

#     def similarity(self, query_vecs, doc_vecs):
#         return (query_vecs * doc_vecs).sum(dim=1)


# class SepBertEncoder(torch.nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
#         self.bert = CustomBertModelWrapper.from_pretrained('bert-base-uncased')
#         self.CLS = self.tokenizer.convert_tokens_to_ids('[CLS]')
#         self.SEP = self.tokenizer.convert_tokens_to_ids('[SEP]')
#         self.bert.set_trainable(True)

#     def lexicon_size(self):
#         return self.tokenizer._tokenizer.get_vocab_size()

#     def enc_query_doc(self, query_text, doc_text):
#         enc_query = self.tokenizer.batch_encode_plus(query_text, return_tensors='pt', padding=True, truncation=True)
#         enc_query = {k: v[:, :500] for k, v in enc_query.items()}
#         if torch.cuda.is_available():
#             enc_query = {k: v.cuda() for k, v in enc_query.items()}
#         result_query = self.bert(**enc_query)
#         query_tok = enc_query['input_ids'][:, 1:]
#         query_tok[query_tok < 1000] = -1

#         enc_doc = self.tokenizer.batch_encode_plus(doc_text, return_tensors='pt', padding=True, truncation=True)
#         enc_doc = {k: v[:, :500] for k, v in enc_doc.items()}
#         if torch.cuda.is_available():
#             enc_doc = {k: v.cuda() for k, v in enc_doc.items()}
#         enc_doc['token_type_ids'][:, :] = 1
#         result_doc = self.bert(**enc_doc)
#         doc_tok = enc_doc['input_ids'][:, 1:]
#         doc_tok[doc_tok < 1000] = -1

#         return {
#             'query': result_query[-1][:, 1:],
#             'query_cls': result_query[-1][:, 0],
#             'query_tok': query_tok,
#             'query_len': enc_query['attention_mask'].sum(dim=1) - 2,
#             'doc': result_doc[-1][:, 1:],
#             'doc_cls': result_doc[-1][:, 0],
#             'doc_tok': doc_tok,
#             'doc_len': enc_doc['attention_mask'].sum(dim=1) - 2,
#         }


# class CustomBertModelWrapper(BertForPreTraining):
#     def __init__(self, config, depth=None):
#         config.output_hidden_states = True
#         super().__init__(config)
#         self.bert = CustomBertModel(config, depth) # replace with custom model

#     def forward(self, input_ids, token_type_ids, attention_mask):
#         return self.bert(input_ids, token_type_ids, attention_mask)

#     @classmethod
#     def from_pretrained(cls, *args, **kwargs):
#         result = super().from_pretrained(*args, **kwargs)
#         if result.bert.depth is not None:
#             # limit the depth by cutting out layers it doesn't need to calculate
#             result.bert.encoder.layer = result.bert.encoder.layer[:result.bert.depth]
#         else:
#             result.depth = len(result.bert.encoder.layer)
#         return result

#     def set_trainable(self, trainable):
#         for param in self.parameters():
#             param.requires_grad = trainable


# class CustomBertModel(BertPreTrainedModel):
#     """
#     Based on pytorch_pretrained_bert.BertModel, but with some extra goodies:
#      - depth: number of layers to run in BERT, where 0 is the raw embeddings, and -1 is all
#               available layers
#     """
#     def __init__(self, config, depth=None):
#         super(CustomBertModel, self).__init__(config)
#         self.depth = depth
#         self.embeddings = BertEmbeddings(config)
#         self.encoder = BertEncoder(config)
#         self.cls = BertPreTrainingHeads(config)
#         self.apply(self.init_weights)

#     def forward(self, input_ids, token_type_ids, attention_mask):
#         """
#         Based on pytorch_pretrained_bert.BertModel
#         """
#         embedding_output = self.embeddings(input_ids, token_type_ids)
#         if self.depth == 0:
#             return [embedding_output]

#         return self.forward_from_layer(embedding_output, attention_mask)

#     def forward_from_layer(self, embedding_output, attention_mask):
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#         head_mask = [None] * self.config.num_hidden_layers

#         _, encoded_layers = self.encoder(embedding_output, extended_attention_mask, head_mask)
#         return list(encoded_layers)

# def lens2mask(lens, size):
#     mask = []
#     for l in lens.cpu():
#         l = l.item()
#         mask.append(([1] * l) + ([0] * (size - l)))
#     return torch.tensor(mask, device=lens.device).long()
