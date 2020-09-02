"""Independent identical distributed vector of random variables."""
import numpy

from ..baseclass import Dist, declare_stochastic_dependencies
from copy import deepcopy


class Iid(Dist):
    """
    Opaque method for creating independent identical distributed random
    variables from an univariate variable.

    Args:
        dist (Dist):
            Distribution to make into i.i.d. vector. The
            distribution will be copied so to not become a part of
            the dependency graph.
        length (int):
            The length of new distribution.

    Examples:
        >>> distribution = chaospy.Iid(chaospy.Normal(0, 1), 2)
        >>> distribution
        Iid(Normal(mu=0, sigma=1), 2)
        >>> chaospy.Cov(distribution)
        array([[1., 0.],
               [0., 1.]])
        >>> mesh = numpy.mgrid[0.25:0.75:3j, 0.25:0.75:3j].reshape(2, -1)
        >>> mapped_mesh = distribution.inv(mesh)
        >>> mapped_mesh.round(2)
        array([[-0.67, -0.67, -0.67,  0.  ,  0.  ,  0.  ,  0.67,  0.67,  0.67],
               [-0.67,  0.  ,  0.67, -0.67,  0.  ,  0.67, -0.67,  0.  ,  0.67]])
        >>> distribution.fwd(mapped_mesh).round(2)
        array([[0.25, 0.25, 0.25, 0.5 , 0.5 , 0.5 , 0.75, 0.75, 0.75],
               [0.25, 0.5 , 0.75, 0.25, 0.5 , 0.75, 0.25, 0.5 , 0.75]])
        >>> distribution.pdf(mapped_mesh).round(3)
        array([0.101, 0.127, 0.101, 0.127, 0.159, 0.127, 0.101, 0.127, 0.101])
        >>> distribution.sample(4, rule="halton").round(3)
        array([[-1.15 ,  0.319, -0.319,  1.15 ],
               [-0.14 ,  0.765, -0.765,  0.14 ]])
        >>> distribution.mom([[1, 2, 2], [2, 1, 2]]).round(12)
        array([0., 0., 1.])

    """


    def __init__(self, dist, length, rotation=None):
        assert len(dist) == 1 and length >= 1
        self._dependencies = [set([idx]) for idx in declare_stochastic_dependencies(self, length)]
        if rotation is not None:
            self._rotation = list(rotation)
        self._dist = deepcopy(dist)
        self._repr = {"_": [dist, length]}
        Dist.__init__(self)

    def __getitem__(self, index):
        if isinstance(index, int):
            if index >= len(self) or index < -len(self):
                raise IndexError("index out of bounds.")
            dist = deepcopy(self._dist)
            dist._dependencies = [self._dependencies[index].copy()]
            return dist

    def _cdf(self, xloc, cache):
        del cache
        return self._dist.fwd(xloc)

    def _ppf(self, uloc, cache):
        del cache
        return self._dist.inv(uloc)

    def _lower(self, cache):
        del cache
        return numpy.ones(len(self))*self._dist.lower

    def _upper(self, cache):
        del cache
        return numpy.ones(len(self))*self._dist.upper

    def _pdf(self, xloc, cache):
        del cache
        return self._dist.pdf(xloc, decompose=True)

    def _mom(self, kloc, cache):
        del cache
        return numpy.prod([self._dist.mom(k) for k in kloc])

    def _ttr(self, kloc, cache):
        del cache
        return self._dist.ttr(kloc)
