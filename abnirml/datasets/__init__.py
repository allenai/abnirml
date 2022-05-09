import os
import yaml
from ir_datasets.util.download import _DownloadConfig

DownloadConfig = _DownloadConfig(contents=yaml.load(open('./abnirml/etc/downloads.yaml'), Loader=yaml.BaseLoader))

from . import cnn_dailymail
from . import gyafc
from . import jfleg
from . import mspc
from . import nbias
from . import wikiturk
from . import yahoo_l6
from . import xsum
