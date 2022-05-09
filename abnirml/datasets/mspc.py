from collections import namedtuple
from typing import NamedTuple
import io
import contextlib
import ir_datasets
from ir_datasets.indices import PickleLz4FullStore
from . import DownloadConfig
from ir_datasets import Dataset

NAME = 'abnirml:mspc'
BASE_PATH = ir_datasets.util.home_path() / NAME
dlc = DownloadConfig.context(NAME, BASE_PATH)


class MspcDoc(NamedTuple):
    doc_id: str
    text: str
    author: str
    url: str
    agency: str
    date: str
    web_date: str


class MspcDocpair(NamedTuple):
    doc_id_a: str
    doc_id_b: str
    paraphrase: bool


class MspcDocs(ir_datasets.formats.BaseDocs):
    def __init__(self, dlc):
        self.dlc = dlc

    @ir_datasets.util.use_docstore
    def docs_iter(self):
        with self.dlc.stream() as stream:
            for i, line in enumerate(io.TextIOWrapper(stream)):
                if i == 0:
                    continue # header
                cols = line.rstrip().split('\t')
                assert len(cols) == 7
                yield MspcDoc(*cols)

    def docs_cls(self):
        return MspcDoc


    def docs_store(self, field='doc_id'):
        return PickleLz4FullStore(
            path=f'{ir_datasets.util.home_path()/NAME}/docs.pklz4',
            init_iter_fn=self.docs_iter,
            data_cls=self.docs_cls(),
            lookup_field=field,
            index_fields=['doc_id'],
        )

    def docs_count(self):
        return self.docs_store().count()

    def docs_namespace(self):
        return NAME

    def docs_lang(self):
        return 'en'


class MspcDocpairs(ir_datasets.formats.BaseDocPairs):
    def __init__(self, dlcs):
        self.dlcs = dlcs

    def docpairs_iter(self):
        for dlc in self.dlcs:
            with dlc.stream() as stream:
                for i, line in enumerate(io.TextIOWrapper(stream)):
                    if i == 0:
                        continue # header
                    cols = line.rstrip().split('\t')
                    assert len(cols) == 5, cols
                    yield MspcDocpair(cols[1], cols[2], cols[0] == '1')

    def docpairs_cls(self):
        return MspcDocpair


SUBSETS = {}
docs = MspcDocs(dlc['sentences'])
SUBSETS[''] = ir_datasets.datasets.base.Dataset(docs, MspcDocpairs([dlc['pairs/train'], dlc['pairs/test']]))
SUBSETS['train'] = ir_datasets.datasets.base.Dataset(docs, MspcDocpairs([dlc['pairs/train']]))
SUBSETS['test'] = ir_datasets.datasets.base.Dataset(docs, MspcDocpairs([dlc['pairs/test']]))


for s_name, subset in SUBSETS.items():
    if s_name == '':
        ir_datasets.registry.register(NAME, Dataset(subset))
    else:
        ir_datasets.registry.register(f'{NAME}/{s_name}', Dataset(subset))
