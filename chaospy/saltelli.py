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


def Sens_m_sample(poly, dist, samples, rule="R"):

    dim = len(dist)
    Y = Saltelli(dist, samples, poly)

    ones = [1]*dim
    zeros = [0]*dim
    index = [0]*dim

    V = np.var(Y[zeros], -1)

    out = []
    for d in xrange(dim):

        index[d] = 1
        s = np.mean(Y[ones]*(Y[index]-Y[zeros]), -1) / (V+(V==0))*(V!=0)
        out.append(s)
        index[d] = 0

    return np.array(out)


def Sens_t_sample(poly, dist, samples, rule="R"):

    assert isinstance(samples, int)

    dim = len(dist)
    Y = Saltelli(dist, samples, poly)

    zeros = [0]*dim
    index = [1]*dim

    V = np.var(Y[zeros], -1)

    out = []
    for d in xrange(dim):

        index[d] = 0
        s = 1-np.mean((Y[index]-Y[zeros])**2, -1) / (2*V+(V==0))*(V!=0)
        out.append(s)
        index[d] = 1

    return np.array(out)

