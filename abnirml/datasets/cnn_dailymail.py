import tarfile
import io
import os
from collections import namedtuple
import ir_datasets
from ir_datasets.indices import PickleLz4FullStore
from ir_datasets.util import ZipExtract
from ir_datasets import Dataset
from . import DownloadConfig


NAME = 'abnirml:cnn_dailymail'
BASE_PATH = ir_datasets.util.home_path() / NAME
dlc = DownloadConfig.context(NAME, BASE_PATH)


CnnDailyMailDoc = namedtuple('CnnDailyMailDoc', ['doc_id', 'title', 'summary', 'body'])


class CnnDailyMailDocs(ir_datasets.formats.BaseDocs):
    def __init__(self, cnn_stories_dlc, dailymail_stories_dlc, cnn_titles_dlc, dailymail_titles_dlc):
        self.cnn_stories_dlc = cnn_stories_dlc
        self.dailymail_stories_dlc = dailymail_stories_dlc
        self.cnn_titles_dlc = cnn_titles_dlc
        self.dailymail_titles_dlc = dailymail_titles_dlc

    def docs_iter(self):
        return iter(self.docs_store())

    def _docs_iter(self):
        for prefix, stories_dlc, titles_dlc in [('cnn:', self.cnn_stories_dlc, self.cnn_titles_dlc), ('dailymail:', self.dailymail_stories_dlc, self.dailymail_titles_dlc)]:
            titles_byid = {}
            with titles_dlc.stream() as stream:
                for line in io.TextIOWrapper(stream):
                    doc_id, title = line.split('\t')
                    titles_byid[doc_id] = title.strip()
            with stories_dlc.stream() as stream, \
                 tarfile.open(fileobj=stream, mode='r|gz') as tarf:
                for record in tarf:
                    if record.path.endswith('.story'):
                        doc_id = record.path.split('/')[-1].split('.')[0]
                        title = titles_byid[doc_id]
                        doc_id = prefix + doc_id
                        text = tarf.extractfile(record).read().decode()
                        parts = text.split('@highlight')
                        summary = '\n'.join(parts[1:])
                        yield CnnDailyMailDoc(doc_id, title, summary, parts[0].strip())

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
        return CnnDailyMailDoc


CNN_DAILYMAIL_DATASET = Dataset(CnnDailyMailDocs(
    dlc['cnn_stories'],
    dlc['dailymail_stories'],
    dlc['cnn_titles'],
    dlc['dailymail_titles'],
))

ir_datasets.registry.register(NAME, CNN_DAILYMAIL_DATASET)
