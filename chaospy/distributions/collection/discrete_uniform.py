"""Discrete uniform probability distribution."""
import numpy

from ..baseclass import Dist


class DiscreteUniform(Dist):
    """
    Discrete uniform probability distribution.

    Args:
        lower (float, chaospy.Dist):
            Lower threshold of distribution. Must be smaller than ``upper``.
            Value will be rounded up to closes integer.
        upper (float, chaospy.Dist):
            Upper threshold of distribution. Value will be rouned down to
            closes integer.

    Examples:
        >>> distribution = chaospy.DiscreteUniform(2, 4)
        >>> print(distribution)
        DiscreteUniform(lower=2, upper=4)
        >>> q = numpy.linspace(0, 1, 9)
        >>> print(numpy.around(distribution.inv(q), 4))
        [2 2 2 3 3 3 4 4 4]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.  0.  0.  0.5 0.5 0.5 1.  1.  1. ]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]
        >>> print(numpy.around(distribution.sample(4), 4))
        [3 2 4 3]
        >>> print(numpy.around(distribution.mom(1), 4))
        3.0
        >>> print(numpy.around(distribution.ttr([1, 2, 3]), 4))
        [[3.     3.     3.3333]
         [0.6667 0.3333 0.    ]]
    """
    interpret_as_integer = True

    def __init__(self, lower, upper):
        Dist.__init__(self, lower=lower, upper=upper)

    def _cdf(self, x_data, lower, upper):
        """Cumulative distribution function."""
        return ((numpy.floor(x_data)-numpy.ceil(lower))/
                (numpy.floor(upper)-numpy.ceil(lower)))

    def _bnd(self, x_data, lower, upper):
        """Lower and upper bounds."""
        return numpy.ceil(lower), numpy.floor(upper)

    def _pdf(self, x_data, lower, upper):
        """Probability density function."""
        return x_data**0/(numpy.floor(upper)-numpy.ceil(lower))

    def _ppf(self, q_data, lower, upper):
        """Point percentile function."""
        return (numpy.floor(q_data*(
            numpy.floor(upper+1)-numpy.ceil(lower))+numpy.ceil(lower)))

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
