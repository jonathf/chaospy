"""Independent identical distributed constructor."""
import numpy

from ..baseclass import Dist
from .. import evaluation


class Iid(Dist):
    """
    Opaque method for creating independent identical distributed random
    variables from an univariate variable.

    Examples:
        >>> X = chaospy.Normal()
        >>> Y = chaospy.Iid(X, 4)
        >>> chaospy.seed(1000)
        >>> print(Y.sample())
        [ 0.39502989 -1.20032309  1.64760248 -0.04465437]
        """

    def __init__(self, dist, length):
        assert len(dist) == 1 and length >= 1
        Dist.__init__(self, dist=dist, length=length)

    def _pdf(self, xloc, dist, length, cache):
        """
        Probability density function.

        Example:
            >>> print(chaospy.Iid(chaospy.Uniform(), 2).pdf(
            ...     [[0.5, 1.5], [0.5, 0.5]]))
            [1. 0.]
        """
        output = evaluation.evaluate_density(
            dist, xloc.reshape(1, -1)).reshape(length, -1)
        assert xloc.shape == output.shape
        return output

    def _cdf(self, xloc, dist, length, cache):
        """
        Cumulative distribution function.

        Example:
            >>> print(chaospy.Iid(chaospy.Uniform(0, 2), 2).fwd(
            ...     [[0.1, 0.2, 0.3], [0.2, 0.2, 0.3]]))
            [[0.05 0.1  0.15]
             [0.1  0.1  0.15]]
        """
        output = evaluation.evaluate_forward(
            dist, xloc.reshape(1, -1)).reshape(length, -1)
        assert xloc.shape == output.shape
        return output

    def _ppf(self, uloc, dist, length, cache):
        """
        Point percentile function.

        Example:
            >>> print(chaospy.Iid(chaospy.Uniform(0, 2), 2).inv(
            ...     [[0.1, 0.2, 0.3], [0.2, 0.2, 0.3]]))
            [[0.2 0.4 0.6]
             [0.4 0.4 0.6]]
        """
        output = evaluation.evaluate_inverse(
            dist, uloc.reshape(1, -1)).reshape(length, -1)
        assert uloc.shape == output.shape
        return output

    def _bnd(self, xloc, dist, length, cache):
        """
        boundary function.

        Example:
            >>> print(chaospy.Iid(chaospy.Uniform(0, 2), 2).range(
            ...     [[0.1, 0.2, 0.3], [0.2, 0.2, 0.3]]))
            [[[0. 0. 0.]
              [0. 0. 0.]]
            <BLANKLINE>
             [[2. 2. 2.]
              [2. 2. 2.]]]
        """
        lower, upper = evaluation.evaluate_bound(
            dist, xloc.reshape(1, -1))
        lower = lower.reshape(length, -1)
        upper = upper.reshape(length, -1)
        assert lower.shape == xloc.shape, (lower.shape, xloc.shape)
        assert upper.shape == xloc.shape
        return lower, upper

    def _mom(self, k, dist, length, cache):
        """
        Moment generating function.

        Example:
            >>> print(chaospy.Iid(chaospy.Uniform(), 2).mom(
            ...     [[0, 0, 1], [0, 1, 1]]))
            [1.   0.5  0.25]
        """
        return numpy.prod(dist.mom(k), 0)

    def _ttr(self, k, dist, length, cache):
        """
        Three terms recursion generating function.

        Example:
            >>> print(numpy.around(chaospy.Iid(chaospy.Uniform(), 2).ttr(
            ...     [[0, 0, 1], [0, 1, 1]]), 4))
            [[[ 0.5     0.5     0.5   ]
              [-0.     -0.      0.0833]]
            <BLANKLINE>
             [[ 0.5     0.5     0.5   ]
              [-0.      0.0833  0.0833]]]
        """
        return dist.ttr(k)

    def __getitem__(self, i):
        """
        Slicing function.

        Example:
            >>> dist = chaospy.Iid(chaospy.Uniform(), 3)
            >>> print(dist[0])
            Uniform(lower=0, upper=1)
            >>> print(dist[1])
            Uniform(lower=0, upper=1)
            >>> print(dist[:1])
            Iid(Uniform(lower=0, upper=1), 1)
            >>> print(dist[1:])
            Iid(Uniform(lower=0, upper=1), 2)
        """
        if isinstance(i, int):
            if (i >= len(self)) or (i < -len(self)):
                raise IndexError("dist index out of range")
            return self.prm["dist"]
        if isinstance(i, slice):
            start, stop, step = i.start, i.stop, i.step
            if start is None: start = 0
            if stop is None: stop = len(self)
            if step is None: step = 1
            length = (stop - start) // step
            return Iid(self.prm["dist"], length=length)
        raise IndexError("index not recognised.")

    def __len__(self):
        """
        Example:
            >>> len(chaospy.Iid(chaospy.Uniform(), 4))
            4
            >>> len(chaospy.Iid(chaospy.Uniform(), 9))
            9
        """
        return self.prm["length"]

    def __str__(self):
        return (self.__class__.__name__ + "(" + str(self.prm["dist"]) +
                ", " + str(self.prm["length"]) + ")")
