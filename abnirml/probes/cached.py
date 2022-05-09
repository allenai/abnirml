import os
import gzip
import pickle
import lz4.frame
from pathlib import Path
from .base import Probe
import ir_datasets


class CachedProbe(Probe):
    def __init__(self, probe, path):
        self.probe = probe
        self.path = Path(path)

    def pair_symmetry(self):
        return self.probe.pair_symmetry()

    def pairs_iter(self):
        if not self.path.exists():
            with ir_datasets.util.finialized_file(self.path, 'wb') as f:
                with lz4.frame.LZ4FrameFile(f, 'wb') as f:
                    for pair in self.probe.pairs_iter():
                        pickle.dump(pair, f)
                        yield pair
        else:
            with lz4.frame.LZ4FrameFile(self.path, 'rb') as f:
                while f.peek():
                    yield pickle.load(f)

    def cache_exists(self):
        return os.path.exists(self.path) or os.path.exists(f'{self.path}.tmp')
