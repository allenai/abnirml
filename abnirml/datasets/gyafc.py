import os
from collections import namedtuple
import ir_datasets
from ir_datasets.util import ZipExtractCache
from . import DownloadConfig
from ir_datasets import Dataset

NAME = 'abnirml:gyafc'
BASE_PATH = ir_datasets.util.home_path() / NAME
dlc = DownloadConfig.context(NAME, BASE_PATH)


GyafcDoc = namedtuple('GyafcDoc', ['doc_id', 'genre', 'split', 'is_orig_formal', 'formal', 'informal', 'mapped_l6_id'])


class GyafcDocs(ir_datasets.formats.BaseDocs):
    def __init__(self, source_dlc):
        self.source_dlc = source_dlc

    def docs_iter(self):
        base = self.source_dlc.path()
        for src, ref in [('informal', 'formal'), ('formal', 'informal')]:
            for genre in ['Entertainment_Music', 'Family_Relationships']:
                for split in ['tune', 'test']:
                    for iref in range(4):
                        # TODO: etc/gyafc qid mappings... Should this be stored somewhere it can be
                        # downloaded independently of the abnirml repo?
                        with open(base/'GYAFC_Corpus'/genre/split/src, 'rt') as s, \
                             open(base/'GYAFC_Corpus'/genre/split/f'{ref}.ref{iref}', 'rt') as r, \
                             open(os.path.join('etc', 'gyafc', genre, split, f'{src}.qids'), 'rt') as qids:
                            for i, (s_line, r_line, qid) in enumerate(zip(s, r, qids)):
                                s_line = s_line.rstrip()
                                r_line = r_line.rstrip()
                                qid = qid.rstrip()
                                yield GyafcDoc(
                                    f'{src}-{ref}-{genre}-{split}-{i}-{iref}',
                                    genre,
                                    split,
                                    src == 'formal',
                                    s_line if src == 'formal' else r_line,
                                    r_line if src == 'formal' else s_line,
                                    qid,
                                )

    def docs_path(self):
        return self.source_dlc.path()

    def docs_cls(self):
        return GyafcDoc

GYAFC_DATASET = Dataset(GyafcDocs(ZipExtractCache(dlc['docs'], BASE_PATH / 'GYAFC_Corpus')))

ir_datasets.registry.register(NAME, GYAFC_DATASET)
