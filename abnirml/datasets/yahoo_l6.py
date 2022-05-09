from collections import namedtuple
import io
import ir_datasets
from ir_datasets.util import GzipExtract, LocalDownload
from ir_datasets.indices import PickleLz4FullStore
from . import DownloadConfig
from ir_datasets import Dataset


NAME = 'abnirml:yahoo-l6'
BASE_PATH = ir_datasets.util.home_path() / NAME
dlc = DownloadConfig.context(NAME, BASE_PATH)


YahooL6Doc = namedtuple('YahooL6Doc', ['doc_id', 'type', 'subject', 'content'])


class YahooL6Docs(ir_datasets.formats.BaseDocs):
    def __init__(self, dlcs):
        self.dlcs = dlcs

    @ir_datasets.util.use_docstore
    def docs_iter(self):
        doc = ''
        bs4 = ir_datasets.lazy_libs.bs4()
        for streamer in self.dlcs:
            with streamer.stream() as stream:
                for line in io.TextIOWrapper(stream):
                    if doc or '<vespaadd>' in line:
                        doc += line
                    if '</vespaadd>' in line:
                        soup = bs4.BeautifulSoup(doc, 'lxml')
                        topic = soup.find('document')
                        did = topic.find('uri').get_text()
                        content = topic.find('content')
                        yield YahooL6Doc(
                            f'{did}-q',
                            'question',
                            topic.find('subject').get_text().replace('<br />', '\n'),
                            content.get_text().replace('<br />', '\n') if content else '')
                        for i, nba in enumerate(topic.find_all('answer_item')):
                            yield YahooL6Doc(
                                f'{did}-a{i}',
                                'answer',
                                '',
                                nba.get_text().replace('<br />', '\n'))
                        doc = ''

    def docs_path(self):
        return self.dlcs[0].path()

    def docs_store(self, field='doc_id'):
        return PickleLz4FullStore(
            path=f'{ir_datasets.util.home_path()/NAME}/docs.pklz4',
            init_iter_fn=self.docs_iter,
            data_cls=self.docs_cls(),
            lookup_field=field,
            index_fields=['doc_id'],
        )

    def docs_cls(self):
        return YahooL6Doc

base_path = ir_datasets.util.home_path() / 'yahoo-l6'

L6_DATASET = Dataset(YahooL6Docs([
    GzipExtract(dlc['docs1']),
    GzipExtract(dlc['docs2']),
]))

ir_datasets.registry.register(NAME, L6_DATASET)
