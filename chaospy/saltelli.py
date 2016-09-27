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
        for i in range(len(indices)):
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

    yo = Y[ones]
    yz = Y[zeros]
    mean = .5*(np.mean(yo) + np.mean(yz))

    yo -= mean
    yz -= mean

    out = []
    for d in range(dim):

        index[d] = 1

        yi = Y[index]-mean

        s = np.mean(yo*(yi-yz), -1) / (V+(V==0))*(V!=0)
        out.append(s)
        index[d] = 0

    return np.array(out)


def Sens_m2_sample(poly, dist, samples, rule="R"):

    dim = len(dist)

    Y = Saltelli(dist, samples, poly)

    ones = [1]*dim
    zeros = [0]*dim
    index = [0]*dim

    V = np.var(Y[zeros], -1)

    yo = Y[ones]
    yz = Y[zeros]
    mean = .5*(np.mean(yo) + np.mean(yz))

    yo -= mean
    yz -= mean

    out = np.empty((dim,dim)+poly.shape)
    for d1 in range(dim):

        index[d1] = 1
        yi = Y[index]-mean
        s = np.mean(yo*(yi-yz), -1) / (V+(V==0))*(V!=0)
        out[d1,d1] = s

        for d2 in range(d1+1, dim):

            index[d2] = 1

            yi = Y[index]-mean

            s = np.mean(yo*(yi-yz), -1) / (V+(V==0))*(V!=0)
            out[d1,d2] = out[d2,d1] = s

            index[d2] = 0

        index[d1] = 0

    return out


def Sens_t_sample(poly, dist, samples, rule="R"):

    dim = len(dist)
    Y = Saltelli(dist, samples, poly)

    zeros = [0]*dim
    index = [1]*dim

    V = np.var(Y[zeros], -1)

    out = []
    for d in range(dim):

        index[d] = 0
        s = 1-np.mean((Y[index]-Y[zeros])**2, -1) / (2*V+(V==0))*(V!=0)
        out.append(s)
        index[d] = 1

    return np.array(out)

if __name__ == "__main__":

    import __init__ as cp

    rho = 0.5
    poly = cp.basis(0,2,2)
    dist = cp.Iid(cp.Normal(), 2)
    dist_dep = cp.MvNormal([0,0], [[1, rho], [rho, 1]])

    print(poly)
    print("md\n", Sens_m_sample(poly, dist_dep, 10**6))
    print("m2d\n", Sens_m2_sample(poly, dist_dep, 10**6)[0,1])
    print("td\n", Sens_t_sample(poly, dist_dep, 10**6))
    print([[0,1,0,1,1/(1+rho*rho),0], [0,0,1,0,1/(1+rho*rho),1]])
