"""Double Weibull distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class double_weibull(SimpleDistribution):
    """Double weibull distribution."""

    def __init__(self, c):
        super(double_weibull, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        ax = numpy.abs(x)
        Px = c/2.0*ax**(c-1.0)*numpy.exp(-ax**c)
        return Px

    def _cdf(self, x, c):
        Cx1 = 0.5*numpy.exp(-abs(x)**c)
        return numpy.where(x > 0, 1-Cx1, Cx1)

    def _ppf(self, q, c):
        q_ = numpy.where(q > .5, 1-q, q)
        Cq1 = numpy.where(q_ == 0, self._upper(c), 1)
        Cq1 = numpy.where((q != 0) & (c != 0),
                          (-numpy.log(2*q_))**(1./c), Cq1)
        return numpy.where(q > .5, Cq1, -Cq1)

    def _lower(self, c):
        return -(-numpy.log(2e-10))**(1./c)

    def _upper(self, c):
        return (-numpy.log(2e-10))**(1./c)


class DoubleWeibull(ShiftScaleDistribution):
    """
    Double Weibull distribution.

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.DoubleWeibull(2, 4, 2)
        >>> distribution
        DoubleWeibull(2, scale=4, shift=2)
        >>> q = numpy.linspace(0, 1, 5)
        >>> distribution.inv(q).round(4)
        array([-16.903 ,  -1.3302,   2.    ,   5.3302,  20.903 ])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.    , 0.1041, 0.    , 0.1041, 0.    ])
        >>> distribution.sample(4).round(4)
        array([ 4.4232, -2.8491,  8.0772,  1.2382])
        >>> distribution.mom(1).round(4)
        2.0
    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(DoubleWeibull, self).__init__(
            dist=double_weibull(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
