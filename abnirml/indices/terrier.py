import os
import shutil
import json
from pytools import memoize_method
import pyterrier
import ir_datasets
from abnirml.java import J


_logger = ir_datasets.log.easy()



class TerrierIndex:
    def __init__(self, path):
        self._path = path

    def path(self):
        return self._path

    def _index_ref(self):
        return pyterrier.IndexRef.of(os.path.join(self.path(), 'data.properties'))

    @memoize_method
    def _index(self):
        J.initialize()
        return pyterrier.IndexFactory.of(self._index_ref())

    def tokenize(self, text, include_stops=True):
        J.initialize()
        pyterrier.index.run_autoclass()
        tokenizer = pyterrier.index.Tokeniser.getTokeniser()
        stemmer = J._autoclass("org.terrier.terms.PorterStemmer")()
        toks = tokenizer.tokenise(J._autoclass("java.io.StringReader")(text))
        if not include_stops:
            stopwords = J._autoclass("org.terrier.terms.Stopwords")(None)
        result = []
        while toks.hasNext():
            tok = toks.next()
            if not include_stops and stopwords.isStopword(tok):
                continue
            if tok is not None: # for some reason, always ends in None
                result.append(stemmer.stem(tok))
        return result

    def did_lookup(self, internal_id):
        meta_index = self._index().getMetaIndex()
        return meta_index.getItem('docno', internal_id)

    def internal_id_lookup(self, did):
        meta_index = self._index().getMetaIndex()
        return meta_index.getDocument('docno', did)

    def built(self):
        return os.path.exists(os.path.join(self.path(), 'config.json'))

    def build(self, doc_iter, field):
        J.initialize()
        path = self.path()
        path_exists = os.path.exists(os.path.join(path, 'config.json'))
        if not path_exists:
            tmp_path = path + '.tmp'
            # TODO: handle multiple fields
            def _doc_iter():
                dids = set()
                for doc in doc_iter:
                    if doc.doc_id in dids:
                        _logger.warn(f'did {doc.doc_id} already encountered. Ignoring this occurrence.')
                    else:
                        dids.add(doc.doc_id)
                        if field is None:
                            doc_dict = dict(docno=doc.doc_id, **dict(zip(doc._fields[1:], doc[1:])))
                        else:
                            doc_dict = {'docno': doc.doc_id, field: doc[doc._fields.index(field)]}
                        yield doc_dict
            indexer = pyterrier.IterDictIndexer(tmp_path)
            indexer.setProperties(**{'indexer.meta.reverse.keys': 'docno'})
            with _logger.duration('indexing'):
                indexer.index(_doc_iter())
                with open(os.path.join(tmp_path, 'config.json'), 'wt') as f:
                    json.dump({}, f)
                if path_exists:
                    _logger.warn('removing existing index')
                    shutil.rmtree(path)
                os.rename(tmp_path, path)

    def idf(self, term):
        idx = self._index()
        lex = idx.getLexicon()
        if term in lex:
            return 1 / (lex[term].getDocumentFrequency() + 1)
        return 1 / (idx.getCollectionStatistics().getNumberOfDocuments() + 1)

    def doc(self, did):
        return TerrierDoc(self._index(), self.internal_id_lookup(did), did)

    def parse_query(self, query, field=None):
        if isinstance(query, str):
            result = self.parse_query({'or': query}, field)
        else:
            result = self._parse_query(query, field)
        if result.strip() in ('()', ''):
            result = 'a' # dummy query
        return f'applypipeline:off {result}'

    def _parse_query(self, query, field=None):
        if isinstance(query, str):
            return self.tokenize(query)
        if isinstance(query, list):
            return [self._parse_query(q, field) for q in query]
        if 'or' in query:
            result = []
            for subq in self._parse_query(query['or'], field=query.get('field', field)):
                if not isinstance(subq, list):
                    subq = [subq]
                for q in subq:
                    result.append(q)
            result = ' '.join(result)
            result = f'({result})'
            if 'weight' in query:
                result = f'{result}^{query["weight"]}'
            return result
        if 'and' in query:
            result = []
            for subq in self._parse_query(query['and'], field=query.get('field', field)):
                if not isinstance(subq, list):
                    subq = [subq]
                for q in subq:
                    result.append(f'+{q}')
            result = ' '.join(result)
            result = f'({result})'
            if 'weight' in query:
                result = f'{result}^{query["weight"]}'
            return result
        if 'terms' in query:
            result = []
            for subq in self._parse_query(query['terms'], field=query.get('field', field)):
                if not isinstance(subq, list):
                    subq = [subq]
                for q in subq:
                    result.append(q)
            result = ' '.join(result)
            result = f'({result})'
            if 'weight' in query:
                result = f'{result}^{query["weight"]}'
            return result
        if 'seq' in query:
            result = []
            for subq in self._parse_query(query['seq'], field=query.get('field', field)):
                if not isinstance(subq, list):
                    subq = [subq]
                for q in subq:
                    result.append(q)
            result = ' '.join(result)
            result = f'"{result}"'
            if 'slop' in query:
                result = f'{result}~{query["slop"]}'
            if 'weight' in query:
                result = f'{result}^{query["weight"]}'
            return result
        raise ValueError(query)


class TerrierDoc:
    def __init__(self, terrier_index, internal_id, did):
        self.did = did
        self.internal_id = internal_id
        self.terrier_index = terrier_index

    def __len__(self):
        if self.internal_id == -1: # not found
            return 0
        doi = self.terrier_index.getDocumentIndex()
        return doi.getDocumentLength(self.internal_id)

    def tfs(self, terms=None):
        di = self.terrier_index.getDirectIndex()
        doi = self.terrier_index.getDocumentIndex()
        lex = self.terrier_index.getLexicon()
        if terms is not None:
            term_id_map = {}
            for term in terms:
                if term in lex:
                    term_id_map[lex[term].getTermId()] = term
            result = {term: 0 for term in terms}
        else:
            term_id_map = {}
            result = {}
        try:
            for posting in di.getPostings(doi.getDocumentEntry(self.internal_id)):
                termid = posting.getId()
                if terms is not None:
                    if termid in term_id_map:
                        result[term_id_map[termid]] = posting.getFrequency()
                else:
                    lee = lex.getLexiconEntry(termid)
                    result[lee.getKey()] = posting.getFrequency()
        except J.JavaException as e:
            _logger.warn(f'Unable to get tfs for did={self.did}: {e}')
        return result


class PtIndexWrapper:
    def __new__(cls, dataset, *args, **kwargs):
        # don't re-wrap
        if hasattr(dataset, '_ptindex'):
            return dataset
        result = super().__new__(cls)
        result.__init__(dataset, *args, **kwargs)
        return result

    def __init__(self, dataset, field=None): # None means all
        self._dataset = dataset
        self._ptindex = None
        self._field = field

    def __getattr__(self, attr):
        return getattr(self._dataset, attr)

    def docs_ptindex(self):
        if self._ptindex is None:
            if self._field:
                idx = TerrierIndex(f'{self._dataset.docs_path()}.terrier.{self._field}')
            else:
                idx = TerrierIndex(f'{self._dataset.docs_path()}.terrier')
            if not idx.built():
                doc_iter = self._dataset.docs_iter()
                doc_iter = _logger.pbar(doc_iter, 'building terrier index')
                idx.build(doc_iter, self._field)
            self._ptindex = idx
        return self._ptindex
