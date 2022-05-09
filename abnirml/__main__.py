import traceback
import os
import fcntl
from fnmatch import fnmatch
import json
import ir_datasets
import abnirml


_logger = ir_datasets.log.easy()


class Locker:
    def __init__(self, file):
        self.file = file
        self.fp = None

    def __enter__ (self):
        self.fp = open(self.file, 'w')
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__(self, exc_type, exc_val, exc_tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()

def main_cache_probes(args):
    for aname, axiom in abnirml.PROBES.items():
        if args.probe is not None and not fnmatch(aname, args.probe):
            continue
        if not axiom.cache_exists():
            with _logger.duration(aname):
                for _ in axiom.pairs_iter():
                    pass

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', nargs='+')
    parser.add_argument('--scorer', nargs='+')
    parser.add_argument('--cache_probes', action='store_true')
    args = parser.parse_args()

    if args.cache_probes:
        return main_cache_probes(args)

    cache = {}
    if os.path.exists('results.jsonl'):
        with open('results.jsonl', 'r+t') as result_file:
            for line in result_file:
                if not line.startswith('#'):
                    record = json.loads(line)
                    cache[record['probe'], record['scorer']] = record


    for sname, scorer in abnirml.SCORERS.items():
        if args.scorer is not None and not any(fnmatch(sname, s) for s in args.scorer):
            continue
        for aname, axiom in abnirml.PROBES.items():
            if args.probe is not None and not any(fnmatch(aname, p) for p in args.probe):
                continue
            key = (aname, sname)
            if key not in cache:
                with _logger.duration(key):
                    try:
                        result = abnirml.ev.ProbeExperiment(scorer, axiom)
                        result['probe'] = aname
                        result['scorer'] = sname
                        _logger.info(key)
                        _logger.info(result)
                        cache[key] = result
                        with Locker('results.jsonl.lock'):
                            with open('results.jsonl', 'at') as result_file:
                                json.dump(result, result_file)
                                result_file.write('\n')
                    except Exception as ex:
                        traceback.print_exc()

if __name__ == '__main__':
    main()
