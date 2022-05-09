import io
import os
from collections import namedtuple
import ir_datasets
from ir_datasets.util import ZipExtract
from ir_datasets import Dataset
from . import DownloadConfig


NAME = 'abnirml:nbias'
BASE_PATH = ir_datasets.util.home_path() / NAME
dlc = DownloadConfig.context(NAME, BASE_PATH)


NBiasDoc = namedtuple('NBiasDoc', ['doc_id', 'biased_text', 'neutral_text'])


class NBiasDocs(ir_datasets.formats.BaseDocs):
    def __init__(self, source_dlc):
        self.source_dlc = source_dlc

    def docs_iter(self):
        with self.source_dlc.stream() as s:
            for line in io.TextIOWrapper(s):
                cols = line.split('\t')
                yield NBiasDoc(cols[0], cols[3], cols[4])

    def docs_path(self):
        return self.source_dlc.path()

    def docs_cls(self):
        return NBiasDoc


NBIAS_DATASET = Dataset(NBiasDocs(ZipExtract(dlc['docs'], 'bias_data/WNC/biased.full')))

ir_datasets.registry.register(NAME, NBIAS_DATASET)
