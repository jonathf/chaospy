"""
Constructing a multivariate probability distribution can be done using the
:func:`~chaospy.distributions.operators.joint.J` constructor. E.g.::

    >>> distribution = chaospy.J(
    ...     chaospy.Normal(0, 1), chaospy.Uniform(0, 1))

The created multivariate distribution behaves much like the univariate case::

    >>> mesh = numpy.meshgrid(
    ...     numpy.linspace(0.25, 0.75, 3),
    ...     numpy.linspace(0.25, 0.75, 3),
    ... )
    >>> print(numpy.around(distribution.cdf(mesh), 4))
    [[0.1497 0.1729 0.1933]
     [0.2994 0.3457 0.3867]
     [0.449  0.5186 0.58  ]]
    >>> print(numpy.around(distribution.pdf(mesh), 4))
    [[0.3867 0.3521 0.3011]
     [0.3867 0.3521 0.3011]
     [0.3867 0.3521 0.3011]]
    >>> print(numpy.around(distribution.sample(
    ...     size=6, rule="H", antithetic=True), 4))
    [[-1.1503  1.1503 -1.1503  1.1503  0.3186 -0.3186]
     [ 0.4444  0.4444  0.5556  0.5556  0.7778  0.7778]]
    >>> print(distribution.mom([[2, 4, 6], [1, 2, 3]]))
    [0.5  1.   3.75]
"""
import numpy

from ..baseclass import Dist, StochasticallyDependentError
from .. import evaluation


class J(Dist):
    """
    Joint random variable generator.

    Args:
        args: Dist
            Distribution to join together.
    """

    def __init__(self, *args):
        args = [dist for arg in args
                for dist in (arg if isinstance(arg, J) else [arg])]
        assert all(isinstance(dist, Dist) for dist in args)
        self.inverse_map = {dist: idx for idx, dist in enumerate(args)}
        prm = {"_%03d" % idx: dist for idx, dist in enumerate(args)}
        Dist.__init__(self, **prm)

    def _cdf(self, xloc, cache, **kwargs):
        """
        Examples:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
            >>> print(dist.fwd([[-0.5, 0.5, 1.5], [-1, 0, 1]]))
            [[0.         0.5        1.        ]
             [0.15865525 0.5        0.84134475]]
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> print(dist.fwd([[-0.5, 0.5, 1.5], [0, 1, 2]]))
            [[0.  0.5 1. ]
             [0.5 0.5 0.5]]
        """
        uloc = numpy.zeros(xloc.shape)
        for dist in evaluation.sorted_dependencies(self, reverse=True):
            if dist not in self.inverse_map:
                continue
            idx = self.inverse_map[dist]
            xloc_ = xloc[idx].reshape(1, -1)
            uloc[idx] = evaluation.evaluate_forward(
                dist, xloc_, cache=cache)[0]
        assert uloc.shape == xloc.shape
        return uloc

    def _range(self, xloc, cache):
        """
        Special handle for finding bounds on constrained dists.

        Example:
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> print(dist.range())
            [[0. 0.]
             [1. 2.]]
        """
        uloc = numpy.zeros((2, len(self)))
        for dist in evaluation.sorted_dependencies(self, reverse=True):
            if dist not in self.inverse_map:
                continue
            idx = self.inverse_map[dist]
            xloc_ = xloc[idx].reshape(1, -1)
            uloc[:, idx] = evaluation.evaluate_bound(
                dist, xloc_, cache=cache).flatten()
        return uloc

    def _bnd(self, xloc, cache, **kwargs):
        """
        Example:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
            >>> print(dist.range([[-0.5, 0.5, 1.5], [-1, 0, 1]]))
            [[[ 0.   0.   0. ]
              [-7.5 -7.5 -7.5]]
            <BLANKLINE>
             [[ 1.   1.   1. ]
              [ 7.5  7.5  7.5]]]
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> print(dist.range([[-0.5, 0.5, 1.5], [0, 1, 2]]))
            [[[ 0.   0.   0. ]
              [-0.5  0.5  1.5]]
            <BLANKLINE>
             [[ 1.   1.   1. ]
              [ 0.5  1.5  2.5]]]
        """
        uloc = numpy.zeros((2,)+xloc.shape)
        for dist in evaluation.sorted_dependencies(self, reverse=True):
            if dist not in self.inverse_map:
                continue
            idx = self.inverse_map[dist]
            xloc_ = xloc[idx].reshape(1, -1)
            uloc[:, idx] = evaluation.evaluate_bound(
                dist, xloc_, cache=cache)[:, 0]
            cache[dist] = xloc_
        return uloc

    def _pdf(self, xloc, cache, **kwargs):
        """
        Example:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
            >>> print(numpy.around(dist.pdf([[-0.5, 0.5, 1.5], [-1, 0, 1]]), 4))
            [0.     0.3989 0.    ]
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> print(dist.pdf([[-0.5, 0.5, 1.5], [0, 1, 2]]))
            [0. 1. 0.]
        """
        floc = numpy.zeros(xloc.shape)
        for dist in evaluation.sorted_dependencies(self, reverse=True):
            if dist not in self.inverse_map:
                continue
            idx = self.inverse_map[dist]
            xloc_ = xloc[idx].reshape(1, -1)
            floc[idx] = evaluation.evaluate_density(
                dist, xloc_, cache=cache)[0]
        return floc

    def _ppf(self, qloc, cache, **kwargs):
        """
        Example:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
            >>> print(numpy.around(dist.inv([[0.1, 0.2, 0.3], [0.3, 0.3, 0.4]]), 4))
            [[ 0.1     0.2     0.3   ]
             [-0.5244 -0.5244 -0.2533]]
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> print(numpy.around(dist.inv([[0.1, 0.2, 0.3], [0.3, 0.3, 0.4]]), 4))
            [[0.1 0.2 0.3]
             [0.4 0.5 0.7]]
        """
        xloc = numpy.zeros(qloc.shape)
        for dist in evaluation.sorted_dependencies(self, reverse=True):
            if dist not in self.inverse_map:
                continue
            idx = self.inverse_map[dist]
            qloc_ = qloc[idx].reshape(1, -1)
            xloc[idx] = evaluation.evaluate_inverse(
                dist, qloc_, cache=cache)[0]
        return xloc

    def _mom(self, kloc, cache, **kwargs):
        """
        Example:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
            >>> print(numpy.around(dist.mom([[0, 0, 1], [0, 1, 1]]), 4))
            [1. 0. 0.]
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> print(numpy.around(dist.mom([1, 1]), 4))
            0.5833
        """
        if evaluation.get_dependencies(*list(self.inverse_map)):
            raise StochasticallyDependentError(
                "Joint distribution with dependencies not supported.")
        output = 1.
        for dist in evaluation.sorted_dependencies(self):
            if dist not in self.inverse_map:
                continue
            idx = self.inverse_map[dist]
            kloc_ = kloc[idx].reshape(1)
            output *= evaluation.evaluate_moment(dist, kloc_, cache=cache)
        return output

    def _ttr(self, kloc, cache, **kwargs):
        """
        Example:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal(), chaospy.Exponential())
            >>> print(numpy.around(dist.ttr([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), 4))
            [[[0.5    0.5    0.5   ]
              [0.0833 0.0667 0.0643]
              [0.     0.     0.    ]]
            <BLANKLINE>
             [[1.     2.     3.    ]
              [3.     5.     7.    ]
              [1.     4.     9.    ]]]
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> print(numpy.around(dist.ttr([1, 1]), 4)) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
                ...
            chaospy.distributions.baseclass.StochasticallyDependentError: Joint ...
        """
        if evaluation.get_dependencies(*list(self.inverse_map)):
            raise StochasticallyDependentError(
                "Joint distribution with dependencies not supported.")
        output = numpy.zeros((2,)+kloc.shape)
        for dist in evaluation.sorted_dependencies(self):
            if dist not in self.inverse_map:
                continue
            idx = self.inverse_map[dist]
            kloc_ = kloc[idx].reshape(1)
            values = evaluation.evaluate_recurrence_coefficients(
                dist, kloc_, cache=cache)
            output.T[idx] = values.T
        return output


    def __len__(self):
        return len(self.inverse_map)

    def __str__(self):
        """
        Example:
            >>> print(chaospy.J(chaospy.Uniform(), chaospy.Normal()))
            J(Uniform(lower=0, upper=1), Normal(mu=0, sigma=1))
        """
        args = [str(self.prm[key]) for key in sorted(list(self.prm))]
        return self.__class__.__name__ + "(" + ", ".join(args) + ")"

    def __getitem__(self, i):
        """
        Example:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
            >>> print(dist[0])
            Uniform(lower=0, upper=1)
            >>> print(dist[1])
            Normal(mu=0, sigma=1)
            >>> print(dist[:1])
            J(Uniform(lower=0, upper=1))
            >>> print(dist[:2])
            J(Uniform(lower=0, upper=1), Normal(mu=0, sigma=1))
            >>> print(dist[2])
            Traceback (most recent call last):
                ...
            IndexError: index out of bounds.
        """
        if isinstance(i, int):
            i = "_%03d" % i
            if i in self.prm:
                return self.prm[i]
            raise IndexError("index out of bounds.")
        if isinstance(i, slice):
            start, stop, step = i.start, i.stop, i.step
            if start is None: start = 0
            if stop is None: stop = len(self)
            if step is None: step = 1
            out = []
            prm = self.prm
            for i in range(start, stop, step):
                out.append(prm["_%03d" % i])
            return J(*out)
        raise IndexError("index not recognised.")
