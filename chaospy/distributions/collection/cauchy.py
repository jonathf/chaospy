"""Cauchy distribution."""
import numpy

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class cauchy(SimpleDistribution):
    """Standard Cauchy distribution."""

    def __init__(self):
        super(cauchy, self).__init__()

    def _pdf(self, x):
        return 1.0/numpy.pi/(1.0+x*x)

    def _cdf(self, x):
        return 0.5 + 1.0/numpy.pi*numpy.arctan(x)

    def _ppf(self, q):
        return numpy.tan(numpy.pi*q-numpy.pi/2.0)

    def _lower(self):
        return -3e13

    def _upper(self):
        return 3e13


class Cauchy(ShiftScaleDistribution):
    """
    Cauchy distribution.

    Also known as Lorentz distribution, Cachy-Lorentz distribution, and
    Breit-Wigner distribution.

    Args:
        shift (float, Distribution):
            Location parameter
        scale (float, Distribution):
            Scaling parameter

    Examples:
        >>> distribution = chaospy.Cauchy()
        >>> distribution
        Cauchy()
        >>> uloc = numpy.linspace(0.1, 0.9, 5)
        >>> uloc
        array([0.1, 0.3, 0.5, 0.7, 0.9])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([-3.078, -0.727,  0.   ,  0.727,  3.078])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.03 , 0.208, 0.318, 0.208, 0.03 ])
        >>> distribution.sample(4).round(3)
        array([ 0.524, -2.646,  6.35 , -0.056])

    Notes:
        The Cauchy distribution is what is known as a "pathological"
        distribution. It is not only infinitely bound, but heavy tailed
        enough that approximate bounds is also infinite for any reasonable
        approximation. This makes both bounds and moments results in
        non-sensibel results. E.g.::

            >>> distribution.lower < -1e10
            array([ True])
            >>> distribution.upper > 1e10
            array([ True])

    """

    def __init__(self, scale=1, shift=0):
        super(Cauchy, self).__init__(
            dist=cauchy(),
            scale=scale,
            shift=shift,
        )
