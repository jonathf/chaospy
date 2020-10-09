"""Discrete uniform probability distribution."""
import numpy

from ..baseclass import SimpleDistribution
from ..operators import J


class discrete_uniform(SimpleDistribution):
    """
    Discrete uniform probability distribution.

    Args:
        lower (float, chaospy.Distribution):
            Lower threshold of distribution. Must be smaller than ``upper``.
            Value will be rounded up to closes integer.
        upper (float, chaospy.Distribution):
            Upper threshold of distribution. Value will be rouned down to
            closes integer.

    Examples:
        >>> distribution = chaospy.DiscreteUniform(2, 4)
        >>> distribution
        DiscreteUniform(2, 4)
        >>> q = numpy.linspace(0, 1, 9)
        >>> q.round(2)
        array([0.  , 0.12, 0.25, 0.38, 0.5 , 0.62, 0.75, 0.88, 1.  ])
        >>> distribution.inv(q).round(2)
        array([1.5 , 1.88, 2.25, 2.62, 3.  , 3.38, 3.75, 4.12, 4.5 ])
        >>> distribution.fwd(distribution.inv(q)).round(2)
        array([0.  , 0.12, 0.25, 0.38, 0.5 , 0.62, 0.75, 0.88, 1.  ])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        >>> distribution.sample(4)
        array([3, 2, 4, 3])
        >>> distribution.mom(1).round(4)
        3.0
        >>> distribution.ttr([0, 1, 2, 3]).round(4)
        array([[3.    , 3.    , 3.    , 3.3333],
               [1.    , 0.6667, 0.3333, 0.    ]])

    """
    interpret_as_integer = True

    def __init__(self, lower, upper):
        super(discrete_uniform, self).__init__(dict(lower=lower, upper=upper))

    def _cdf(self, x_data, lower, upper):
        """Cumulative distribution function."""
        lower = numpy.round(lower)
        upper = numpy.round(upper)
        out = (x_data-lower+0.5)/(upper-lower+1)
        return out

    def _lower(self, lower, upper):
        """Lower bounds."""
        return numpy.round(lower)-0.5

    def _upper(self, lower, upper):
        """Upper bounds."""
        return numpy.round(upper)+0.5

    def _pdf(self, x_data, lower, upper):
        """Probability density function."""
        return x_data**0/(numpy.round(upper)-numpy.round(lower))

    def _ppf(self, q_data, lower, upper):
        """Point percentile function."""
        lower = numpy.round(lower)
        upper = numpy.rint(upper)
        return q_data*(upper-lower+1)+lower-0.5

    def _mom(self, k_data, lower, upper):
        """Raw statistical moments."""
        return numpy.mean(numpy.arange(
            numpy.ceil(lower), numpy.floor(upper)+1)**k_data)

    def _ttr(self, k_data, lower, upper):
        """Three terms recurrence coefficients."""
        from chaospy.quadrature import discretized_stieltjes
        abscissas = numpy.arange(numpy.ceil(lower), numpy.floor(upper)+1)
        weights = numpy.repeat(1./len(abscissas), len(abscissas))
        (alpha, beta), _, _ = discretized_stieltjes(k_data, [abscissas], weights)
        return alpha[0, -1], beta[0, -1]


class DiscreteUniform(J):

    interpret_as_integer = True

    def __init__(self, lower, upper):
        super(DiscreteUniform, self).__init__(discrete_uniform(lower=lower, upper=upper))
        self._repr_args = [lower, upper]
