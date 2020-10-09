"""Binomial probability distribution."""
from functools import wraps
import numpy
from scipy import special

from ..baseclass import SimpleDistribution
from ..operators import J


class binomial(SimpleDistribution):
    """
    Binomial probability distribution.

    Point density:
        comb(N, x) p^x (1-p)^{N-x}      x in {0, 1, ..., N}

    Examples:
        >>> distribution = chaospy.Binomial(3, 0.5)
        >>> distribution
        Binomial(3, 0.5)
        >>> distribution.pdf([0, 1, 2, 3]).round(4)
        array([0.125, 0.375, 0.375, 0.125])
        >>> distribution.cdf([0, 1, 2, 3]).round(4)
        array([0.125, 0.5  , 0.875, 1.   ])
        >>> distribution.fwd([-0.5, -0.49, 0, 0.49, 0.5]).round(4)
        array([0.    , 0.0013, 0.0625, 0.1238, 0.125 ])
        >>> q = numpy.linspace(0, 1, 8)
        >>> q.round(2)
        array([0.  , 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1.  ])
        >>> distribution.inv(q).round(2)
        array([-0.5 ,  0.55,  0.93,  1.31,  1.69,  2.07,  2.45,  3.5 ])
        >>> distribution.fwd(distribution.inv(q)).round(2)
        array([0.  , 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1.  ])
        >>> distribution.sample(10)
        array([3, 1, 2, 1, 0, 2, 2, 2, 2, 3])
        >>> distribution.mom([1, 2, 3]).round(4)
        array([1.5 , 3.  , 6.75])
        >>> distribution.ttr([0, 1, 2, 3]).round(4)
        array([[1.5 , 1.5 , 1.5 , 1.5 ],
               [1.  , 0.75, 1.  , 0.75]])

    """
    interpret_as_integer = True

    def __init__(self, size, prob):
        super(binomial, self).__init__(
            parameters=dict(size=size, prob=prob),
            repr_args=[size, prob],
        )

    def _cdf(self, x_data, size, prob):
        size = numpy.round(size)
        x_data = x_data-0.5

        floor = numpy.zeros(x_data.shape)
        indices = x_data >= 0
        floor[indices] = special.bdtr(numpy.floor(x_data[indices]), size, prob)

        ceil = numpy.ones(x_data.shape)
        indices = x_data <= size
        ceil[indices] = special.bdtr(numpy.ceil(x_data[indices]), size, prob)
        ceil[numpy.isnan(ceil)] = 0  # left edge case

        offset = x_data-numpy.floor(x_data)
        out = floor*(1-offset) + ceil*offset
        return out

    def _pdf(self, x_data, size, prob):
        x_data = numpy.round(x_data)
        return special.comb(size, x_data)*prob**x_data*(1-prob)**(size-x_data)

    def _lower(self, size, prob):
        return -0.5

    def _upper(self, size, prob):
        return numpy.round(size)+0.5

    def _mom(self, k_data, size, prob):
        x_data = numpy.arange(int(size)+1, dtype=int)
        return numpy.sum(x_data**k_data*self._pdf(
            x_data, size=numpy.floor(size), prob=prob))

    def _ttr(self, k_data, size, prob):
        """Krawtchouk rule."""
        from chaospy.quadrature import discretized_stieltjes
        abscissas = numpy.arange(0, numpy.floor(size)+1)
        weights = self._pdf(abscissas, size, prob)
        (alpha, beta), _, _ = discretized_stieltjes(k_data, [abscissas], weights)
        return alpha[0, -1], beta[0, -1]


class Binomial(J):

    def __init__(self, size, prob):
        dist = binomial(size, prob)
        super(Binomial, self).__init__(dist)
        self._repr_args = [size, prob]
