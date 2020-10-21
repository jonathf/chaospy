"""Wrapped Cauchy distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class wrapped_cauchy(SimpleDistribution):
    """Wrapped Cauchy distribution."""

    def __init__(self, c):
        super(wrapped_cauchy, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        return (1.0-c*c)/(2*numpy.pi*(1+c*c-2*c*numpy.cos(x)))

    def _cdf(self, x, c):
        output = 0.0*x
        val = (1.0+c)/(1.0-c)
        c1 = x<numpy.pi
        c2 = 1-c1

        xn = numpy.extract(c2,x)
        if (any(xn)):
            valn = numpy.extract(c2, numpy.ones_like(x)*val)
            xn = 2*numpy.pi - xn
            yn = numpy.tan(xn/2.0)
            on = 1.0-1.0/numpy.pi*numpy.arctan(valn*yn)
            numpy.place(output, c2, on)

        xp = numpy.extract(c1,x)
        if (any(xp)):
            valp = numpy.extract(c1, numpy.ones_like(x)*val)
            yp = numpy.tan(xp/2.0)
            op = 1.0/numpy.pi*numpy.arctan(valp*yp)
            numpy.place(output, c1, op)

        return output

    def _ppf(self, qloc, c):
        val = (1.0-c)/(1.0+c)
        rcq = 2*numpy.arctan(val*numpy.tan(numpy.pi*qloc))
        rcmq = 2*numpy.pi-2*numpy.arctan(val*numpy.tan(numpy.pi*(1-qloc)))
        return numpy.where(qloc < 0.5, rcq, rcmq)

    def _lower(self, c):
        return 0.

    def _upper(self, c):
        return 2*numpy.pi


class WrappedCauchy(ShiftScaleDistribution):
    """
    Wrapped Cauchy distribution

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.WrappedCauchy(0.5)
        >>> distribution
        WrappedCauchy(0.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.475, 1.596, 4.687, 5.808, 6.283])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.477, 0.331, 0.094, 0.094, 0.331, 0.477])
        >>> distribution.sample(4).round(3)
        array([5.15 , 0.251, 6.178, 2.809])

    """

    def __init__(self, shape=0.5, scale=1, shift=0):
        super(WrappedCauchy, self).__init__(
            dist=wrapped_cauchy(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
