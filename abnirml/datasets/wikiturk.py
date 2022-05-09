from typing import NamedTuple, Tuple
import io
import contextlib
import ir_datasets
from ir_datasets.indices import PickleLz4FullStore
from . import DownloadConfig
from ir_datasets import Dataset

# Text simplification dataset from <https://github.com/cocoxu/simplification>
# Wei Xu and Courtney Napoles and Ellie Pavlick and Quanze Chen and Chris Callison-Burch.
# Optimizing Statistical Machine Translation for Text Simplification. TACL 2016.


NAME = 'abnirml:wikiturk'
BASE_PATH = ir_datasets.util.home_path() / NAME
dlc = DownloadConfig.context(NAME, BASE_PATH)


class WikiTurkDoc(NamedTuple):
    doc_id: str
    source: str
    simplifications: Tuple[str, ...]


class WikiTurkDocs(ir_datasets.formats.BaseDocs):
    def __init__(self, source_dlc, simp_dlcs, did_prefix):
        self._source_dlc = source_dlc
        self._simp_dlcs = simp_dlcs
        self._did_prefix = did_prefix

    @ir_datasets.util.use_docstore
    def docs_iter(self):
        with contextlib.ExitStack() as stack:
            src = io.TextIOWrapper(stack.enter_context(self._source_dlc.stream()))
            simps = [io.TextIOWrapper(stack.enter_context(s.stream())) for s in self._simp_dlcs]
            for i, texts in enumerate(zip(src, *simps)):
                texts = [t.strip().replace('-lrb-', '(').replace('-rrb-', ')') for t in texts]
                yield WikiTurkDoc(f'{self._did_prefix}{i}', texts[0], tuple(texts[1:]))

    def docs_cls(self):
        return WikiTurkDoc

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


SUBSETS = {}
SUBSETS['test'] = ir_datasets.datasets.base.Dataset(
    WikiTurkDocs(dlc['test.norm'], [dlc[f'test.{i}'] for i in range(8)], 'test'),
)
SUBSETS['tune'] = ir_datasets.datasets.base.Dataset(
    WikiTurkDocs(dlc['tune.norm'], [dlc[f'tune.{i}'] for i in range(8)], 'tune'),
)
SUBSETS[''] = ir_datasets.datasets.base.Concat(SUBSETS['test'], SUBSETS['tune'])


for s_name, subset in SUBSETS.items():
    if s_name == '':
        ir_datasets.registry.register(NAME, Dataset(subset))
    else:
        ir_datasets.registry.register(f'{NAME}/{s_name}', Dataset(subset))
