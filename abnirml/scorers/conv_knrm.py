import os
import re
import pickle
import hashlib
from pathlib import Path
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from nltk import word_tokenize
import ir_datasets
from .base import NeuralScorer


_logger = ir_datasets.log.easy()


class ConvKNRM(NeuralScorer):
    def __init__(self, weight_path, batch_size=8):
        super().__init__(None, batch_size)
        self.weight_path = weight_path

    def delta(self):
        return 0.434799 # 50th percentile delta from msmarco dev

    def build(self):
        if self.model is None:
            with _logger.duration('loading ConvKNRM model'):
                self.model = ConvKNRMModel()
                self.model.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu') ), strict=False)
                if torch.cuda.is_available():
                    self.model.cuda()


class KNRM(NeuralScorer):
    def __init__(self, weight_path, batch_size=8):
        super().__init__(None, batch_size)
        self.weight_path = weight_path

    def delta(self):
        return 0.31069 # 50th percentile delta from msmarco dev

    def build(self):
        if self.model is None:
            with _logger.duration('loading KNRM model'):
                self.model = KNRMModel()
                self.model.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu') ), strict=False)
                if torch.cuda.is_available():
                    self.model.cuda()


class ConvKNRMModel(torch.nn.Module):
    """
    Implementation of the ConvKNRM model from:
      > Zhuyun Dai, Chenyan Xiong, Jamie Callan, and Zhiyuan Liu. 2018. Convolutional Neural
      > Networks for Soft-Matching N-Grams in Ad-hoc Search. In WSDM.
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = Tokenizer()
        self.embed = self.tokenizer.encoder()
        self.simmat = InteractionMatrix()
        self.padding, self.convs = nn.ModuleList(), nn.ModuleList()
        for conv_size in range(1, 4):
            if conv_size > 1:
                self.padding.append(nn.ConstantPad1d((0, conv_size-1), 0))
            else:
                self.padding.append(nn.Sequential()) # identity
            self.convs.append(nn.ModuleList())
            self.convs[-1].append(nn.Conv1d(self.embed.dim(), 128, conv_size))
        self.kernels = RbfKernelBank.from_strs('-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1.0', '0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.001', dim=1, requires_grad=True)
        channels = 3 ** 2
        self.combine = nn.Linear(self.kernels.count() * channels, 1)

    def forward(self, query_text, doc_text):
        inputs = self.tokenizer.tokenize_queries_docs(query_text, doc_text)
        enc = self.embed.enc_query_doc(**inputs)
        a_embed, b_embed = enc['query'], enc['doc']
        a_reps, b_reps = [], []
        for pad, conv in zip(self.padding, self.convs):
            a_reps.append(conv[0](pad(a_embed.permute(0, 2, 1))).permute(0, 2, 1))
            b_reps.append(conv[0](pad(b_embed.permute(0, 2, 1))).permute(0, 2, 1))
        simmats = []
        for a_rep in a_reps:
            for b_rep in b_reps:
                simmats.append(self.simmat(a_rep, b_rep, inputs['query_tok'], inputs['doc_tok']))
        simmats = torch.cat(simmats, dim=1)
        mask = (simmats != 0.).unsqueeze(1) # which cells are not padding?
        kernels = self.kernels(simmats)
        kernels[~mask.expand(kernels.shape)] = 0. # zero out padding
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        result = kernels.sum(dim=3) # sum over document
        simmats = simmats.reshape(BATCH, 1, VIEWS, QLEN, DLEN) \
                        .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN) \
                        .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        result = torch.where(mask[:,:,0].sum(dim=-1) != 0, (result + 1e-6).log(), torch.zeros_like(result))
        result = result.sum(dim=2) # sum over query terms
        result = self.combine(result)
        return result.flatten()


class KNRMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = Tokenizer()
        self.encoder = self.tokenizer.encoder()
        self.simmat = InteractionMatrix()
        self.kernels = RbfKernelBank.from_strs('-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1.0', '0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.001', dim=1, requires_grad=True)
        self.combine = nn.Linear(self.kernels.count(), 1)

    def forward(self, query_text, doc_text):
        inputs = self.tokenizer.tokenize_queries_docs(query_text, doc_text)
        simmat = self.simmat.encode_query_doc(self.encoder, **inputs)
        kernel_scores = self.kernel_pool(simmat)
        result = self.combine(kernel_scores) # linear combination over kernels
        return result.flatten()

    def kernel_pool(self, simmat):
        mask = (simmat != 0.).unsqueeze(1) # which cells are not padding?
        kernels = self.kernels(simmat)
        kernels[~mask.expand(kernels.shape)] = 0. # zero out padding
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        result = kernels.sum(dim=3) # sum over document
        simmat = simmat.reshape(BATCH, 1, VIEWS, QLEN, DLEN) \
                       .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN) \
                       .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        result = torch.where(mask[:,0].sum(dim=3) != 0, (result + 1e-6).log(), torch.zeros_like(result))
        result = result.sum(dim=2) # sum over query terms
        return result


class RbfKernelBank(nn.Module):
    def __init__(self, mus=None, sigmas=None, dim=0, requires_grad=True):
        super().__init__()
        self.mus = nn.Parameter(torch.tensor(mus), requires_grad=requires_grad)
        self.sigmas = nn.Parameter(torch.tensor(sigmas), requires_grad=requires_grad)
        self.dim = dim

    def forward(self, data):
        shape = list(data.shape)
        shape.insert(self.dim, 1)
        data = data.reshape(*shape)
        shape = [1]*len(data.shape)
        shape[self.dim] = -1
        mus, sigmas = self.mus.reshape(*shape), self.sigmas.reshape(*shape)
        adj = data - mus
        return torch.exp(-0.5 * adj * adj / sigmas / sigmas)

    def count(self):
        return self.mus.shape[0]

    @staticmethod
    def from_strs(mus='-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1.0', \
        sigmas='0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.001', dim=-1, requires_grad=True):
        mus = [float(x) for x in mus.split(',')]
        sigmas = [float(x) for x in sigmas.split(',')]
        return RbfKernelBank(mus, sigmas, dim=dim, requires_grad=requires_grad)

    @staticmethod
    def evenly_spaced(count=11, sigma=0.1, rng=(-1, 1), dim=-1, requires_grad=True):
        mus = [x.item() for x in torch.linspace(rng[0], rng[1], steps=count)]
        sigmas = [sigma for _ in mus]
        return RbfKernelBank(mus, sigmas, dim=dim, requires_grad=requires_grad)

def binmat(a, b, padding=None):
    BAT, A, B = a.shape[0], a.shape[1], b.shape[1]
    a = a.reshape(BAT, A, 1)
    b = b.reshape(BAT, 1, B)
    result = (a == b)
    if padding is not None:
        result = result & (a != padding) & (b != padding)
    return result.float()


def cos_simmat(a, b, amask=None, bmask=None):
    BAT, A, B = a.shape[0], a.shape[1], b.shape[1]
    a_denom = a.norm(p=2, dim=2).reshape(BAT, A, 1) + 1e-9 # avoid 0div
    b_denom = b.norm(p=2, dim=2).reshape(BAT, 1, B) + 1e-9 # avoid 0div
    result = a.bmm(b.permute(0, 2, 1)) / (a_denom * b_denom)
    if amask is not None:
        result = result * amask.reshape(BAT, A, 1)
    if bmask is not None:
        result = result * bmask.reshape(BAT, 1, B)
    return result


class InteractionMatrix(nn.Module):

    def __init__(self, padding=-1):
        super().__init__()
        self.padding = padding

    def forward(self, a_embed, b_embed, a_tok, b_tok):
        wrap_list = lambda x: x if isinstance(x, list) else [x]

        a_embed = wrap_list(a_embed)
        b_embed = wrap_list(b_embed)

        BAT, A, B = a_embed[0].shape[0], a_embed[0].shape[1], b_embed[0].shape[1]

        simmats = []

        for a_emb, b_emb in zip(a_embed, b_embed):
            if a_emb.dtype is torch.long and len(a_emb.shape) == 2 and \
               b_emb.dtype is torch.long and len(b_emb.shape) == 2:
                # binary matrix
                simmats.append(binmat(a_emb, b_emb, padding=self.padding))
            else:
                # cosine similarity matrix
                a_mask = (a_tok.reshape(BAT, A, 1) != self.padding).float()
                b_mask = (b_tok.reshape(BAT, 1, B) != self.padding).float()
                simmats.append(cos_simmat(a_emb, b_emb, a_mask, b_mask))
        return torch.stack(simmats, dim=1)

    def encode_query_doc(self, encoder, **inputs):
        enc = encoder.enc_query_doc(**inputs)
        return self(enc['query'], enc['doc'], inputs['query_tok'], inputs['doc_tok'])












class Tokenizer:
    def __init__(self):
        with open('bing.pkl', 'rb') as f:
            self._terms, self._weights = pickle.load(f)
        self._term2idx = {t: i for i, t in enumerate(self._terms)}
        self._hashspace = 1000
        random = np.random.RandomState(42)
        hash_weights = random.normal(scale=0.5, size=(self._hashspace, self._weights.shape[1]))
        self._weights = np.concatenate([self._weights, hash_weights])

    def tokenize(self, text):
        """
        Meant to be overwritten in to provide vocab-specific tokenization when necessary
        e.g., BERT's WordPiece tokenization
        """
        text = text.lower()
        text = re.sub(r'[^a-z0-9]', ' ', text)
        return text.split()

    def id2tok(self, idx):
        return self._terms[idx]

    def tokenize_queries_docs(self, queries, docs):
        queries = [[self.tok2id(t) for t in self.tokenize(q)] for q in queries]
        docs = [[self.tok2id(t) for t in self.tokenize(d)] for d in docs]
        query_len = [len(q) for q in queries]
        max_q = max(query_len)
        queries = [q + [-1] * (max_q - len(q)) for q in queries]
        doc_len = [len(d) for d in docs]
        max_d = max(doc_len)
        docs = [d + [-1] * (max_d - len(d)) for d in docs]
        result = {'query_tok': torch.tensor(queries), 'query_len': torch.tensor(query_len),
                  'doc_tok': torch.tensor(docs), 'doc_len': torch.tensor(doc_len)}
        if torch.cuda.is_available():
            result = {k: v.cuda() for k, v in result.items()}
        return result

    def lexicon_size(self) -> int:
        return len(self._terms)

    def tok2id(self, tok):
        try:
            return self._term2idx[tok]
        except KeyError:
            # NOTE: use md5 hash (or similar) here because hash() is not consistent across runs
            item = tok.encode()
            item_hash = int(hashlib.md5(item).hexdigest(), 16)
            item_hash_pos = item_hash % self._hashspace
            return len(self._terms) + item_hash_pos

    def encoder(self):
        return WordvecEncoder(self)


class WordvecEncoder(nn.Module):

    def __init__(self, vocabulary):
        super().__init__()
        matrix = vocabulary._weights
        self.size = matrix.shape[1]
        matrix = np.concatenate([np.zeros((1, self.size)), matrix]) # add padding record (-1)
        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(matrix.astype(np.float32)))

    def forward(self, toks, lens=None):
        # lens ignored
        return self.embed(toks + 1) # +1 to handle padding at position -1

    def enc_query_doc(self, query_tok, query_len, doc_tok, doc_len):
        return {
            'query': self.forward(query_tok, query_len),
            'doc': self.forward(doc_tok, doc_len),
        }

    def dim(self):
        return self.embed.weight.shape[1]
