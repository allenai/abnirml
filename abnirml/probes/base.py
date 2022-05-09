class Probe:
    def pair_symmetry(self):
        return 'asymmetric' # most probes are asymmetric

    def pairs_iter(self):
        raise NotImplementedError
