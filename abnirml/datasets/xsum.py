import tarfile
import io
import os
from collections import namedtuple
import ir_datasets
from ir_datasets.indices import PickleLz4FullStore
from ir_datasets.util import ZipExtract
from . import DownloadConfig
from ir_datasets import Dataset


NAME = 'abnirml:xsum'
BASE_PATH = ir_datasets.util.home_path() / NAME
dlc = DownloadConfig.context(NAME, BASE_PATH)


XSumDoc = namedtuple('XSumDoc', ['doc_id', 'url', 'title', 'first_sentence', 'rest_body'])


class XSumDocs(ir_datasets.formats.BaseDocs):
    def __init__(self, source_dlc):
        self.source_dlc = source_dlc

    def docs_iter(self):
        return iter(self.docs_store())

    def _docs_iter(self):
        with self.source_dlc.stream() as stream, \
             tarfile.open(fileobj=stream, mode='r|gz') as tarf:
            for record in tarf:
                if record.path.endswith('.summary'):
                    doc_id = record.path.split('/')[-1].split('.')[0]
                    text = tarf.extractfile(record).read().decode()
                    parts = text.split('[SN]')
                    yield XSumDoc(doc_id, parts[2].strip(), parts[4].strip(), parts[6].strip(), parts[8].strip())

    def docs_path(self):
        return self.source_dlc.path()

    def docs_store(self, field='doc_id'):
        return PickleLz4FullStore(
            path=f'{ir_datasets.util.home_path()/NAME}/docs.pklz4',
            init_iter_fn=self._docs_iter,
            data_cls=self.docs_cls(),
            lookup_field=field,
            index_fields=['doc_id'],
        )

    def docs_cls(self):
        return XSumDoc


XSUM_DATASET = Dataset(XSumDocs(dlc['docs']))

ir_datasets.registry.register(NAME, XSUM_DATASET)
