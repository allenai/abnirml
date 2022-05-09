from collections import namedtuple
import io
import contextlib
import ir_datasets
from . import DownloadConfig
from ir_datasets import Dataset

NAME = 'abnirml:jfleg'
BASE_PATH = ir_datasets.util.home_path() / NAME
dlc = DownloadConfig.context(NAME, BASE_PATH)


JflegDoc = namedtuple('JflegDoc', ['doc_id', 'nonfluent', 'fluents'])


class JflegDocs(ir_datasets.formats.BaseDocs):
    def __init__(self, src_dlc, refs_dlc):
        self.src_dlc = src_dlc
        self.refs_dlc = refs_dlc

    def docs_iter(self):
        with contextlib.ExitStack() as ctxt:
            src = io.TextIOWrapper(ctxt.enter_context(self.src_dlc.stream()))
            refs = [io.TextIOWrapper(ctxt.enter_context(r.stream())) for r in self.refs_dlc]
            for i, items in enumerate(zip(src, *refs)):
                nonfluent, *fluents = items
                yield JflegDoc(str(i), nonfluent, tuple(fluents))

    def docs_cls(self):
        return JflegDoc

SUBSETS = {}
SUBSETS['dev'] = JflegDocs(dlc['dev/src'], [dlc['dev/ref0'], dlc['dev/ref1'], dlc['dev/ref2'], dlc['dev/ref3']])
SUBSETS['dev/sp'] = JflegDocs(dlc['dev/src.sp'], [dlc['dev/ref0'], dlc['dev/ref1'], dlc['dev/ref2'], dlc['dev/ref3']])
SUBSETS['test'] = JflegDocs(dlc['test/src'], [dlc['test/ref0'], dlc['test/ref1'], dlc['test/ref2'], dlc['test/ref3']])
SUBSETS['test/sp'] = JflegDocs(dlc['test/src.sp'], [dlc['test/ref0'], dlc['test/ref1'], dlc['test/ref2'], dlc['test/ref3']])
SUBSETS['all'] = ir_datasets.datasets.base.Concat(SUBSETS['dev'], SUBSETS['test'])
SUBSETS['sp'] = ir_datasets.datasets.base.Concat(SUBSETS['dev/sp'], SUBSETS['test/sp'])

for s_name, subset in SUBSETS.items():
    if s_name == 'all':
        ir_datasets.registry.register(NAME, Dataset(subset))
    else:
        ir_datasets.registry.register(f'{NAME}/{s_name}', Dataset(subset))
