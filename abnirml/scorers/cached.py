from pathlib import Path
import gzip
import zlib
import itertools
import hashlib
import pickle
import ir_datasets


_logger = ir_datasets.log.easy()


class SimpleFsCache:
    def __init__(self, path, open_fn=open):
        self._memcache = {}
        if Path(path).exists():
            with open_fn(path, 'r+b') as f:
                underlying_f = f.fileobj if isinstance(f, gzip.GzipFile) else f
                last_valid_pos = None
                while True:
                    try:
                        pos = underlying_f.tell()
                        key, value = pickle.load(f), pickle.load(f)
                        if isinstance(key, list) and isinstance(value, list):
                            for k, v in zip(key, value):
                                self._memcache[k] = v
                        else:
                            self._memcache[key] = value
                        last_valid_pos = pos
                    except EOFError:
                        break # done reading cache
                    except zlib.error as e:
                        # hmmmm, problem decoding.
                        underlying_f.seek(last_valid_pos)
                        underlying_f.truncate()
                        _logger.warn(f'gzip error loading: {repr(e)}. Tuncating to last valid position in file.')
                        break
            _logger.info(f'loaded cache with {len(self._memcache)} items')
        self._file = open_fn(path, 'ab')

    def __contains__(self, key):
        return key in self._memcache

    def __getitem__(self, key):
        return self._memcache[key]

    def __setitem__(self, key, value):
        # avoid bloat by re-saving same value (common when things are be batched)
        if key not in self._memcache or self._memcache[key] != value:
            pickle.dump(key, self._file)
            pickle.dump(value, self._file)
            self._file.flush()
            self._memcache[key] = value


class CachedScorer:
    def __init__(self, scorer, cache_path, hashfn=hashlib.md5):
        self._scorer = scorer
        self._cache_path = Path(cache_path)
        self._cache = None
        self._hashfn = hashfn

    def delta(self):
        return self._scorer.delta()

    def cache(self):
        if self._cache is None:
            self._cache = SimpleFsCache(self._cache_path)
        return self._cache

    def score_iter(self, it):
        cache = self.cache()
        will_be_cached = set() # items that will enter the cache. This may not seem like it does anything, but due to batched scoring, it does
        def cache_lookup_iter(it):
            for qtext, dtext in it:
                key = (qtext, dtext)
                key = self._hashfn(repr(key).encode()).digest()
                if key in cache or key in will_be_cached:
                    yield qtext, dtext, key, False # should not score
                else:
                    will_be_cached.add(key)
                    yield qtext, dtext, key, True # should score

        it = cache_lookup_iter(it)
        it1, it2 = itertools.tee(it, 2)
        def to_be_scored_iter(it):
            for qtext, dtext, _, should_score in it:
                if should_score:
                    yield qtext, dtext
        it2 = self._scorer.score_iter(to_be_scored_iter(it2))

        for _, _, key, should_score in it1:
            if should_score:
                score = next(it2, StopIteration)
                if score is StopIteration:
                    break
                cache[key] = score
                will_be_cached.discard(key) # it's been cached!
            yield cache[key]
