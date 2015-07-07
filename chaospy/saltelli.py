"""
The Saltelli method

Code is built upon the code provided by Vinzenze Eck
"""
import numpy as np

class Saltelli:

    def __init__(self, dist, samples, poly=None, rule="R"):
        self.dist = dist
        samples_ = dist.sample(2*samples, rule=rule)
        self.samples1 = samples_.T[:samples].T
        self.samples2 = samples_.T[samples:].T
        self.poly = poly

    def __getitem__(self, indices):
        assert len(self.dist) == len(indices)

        key = "".join([i and "a" or "b" for i in indices])

        if hasattr(self, key):
            return getattr(self, key)

        new = np.empty(self.samples1.shape)
        for i in xrange(len(indices)):
            if indices[i]:
                new[i] = self.samples1[i]
            else:
                new[i] = self.samples2[i]

        if self.poly:
            new = self.poly(*new)

        setattr(self, key, new)
        return new
