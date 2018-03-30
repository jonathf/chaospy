"""Wrapped Cauchy distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class wrapped_cauchy(Dist):
    """Wrapped Cauchy distribution."""

    def __init__(self, c):
        Dist.__init__(self, c=c)

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

    def _ppf(self, q, c):
        val = (1.0-c)/(1.0+c)
        rcq = 2*numpy.arctan(val*numpy.tan(numpy.pi*q))
        rcmq = 2*numpy.pi-2*numpy.arctan(val*numpy.tan(numpy.pi*(1-q)))
        return numpy.where(q < 1.0/2, rcq, rcmq)

    def _bnd(self, c):
        return 0.0, 2*numpy.pi


class WrappedCauchy(Add):
    """
    Wrapped Cauchy distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter

    Examples:
        >>> distribution = chaospy.WrappedCauchy(0.8, 4, 6)
        >>> print(distribution)
        WrappedCauchy(scale=4, shape=0.8, shift=6)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [ 6.5125  7.521  18.5664 29.6117 30.6202]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.1667 0.3333 0.5    0.6667 0.8333]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.2697 0.0928 0.0044 0.0928 0.2697]
        >>> print(numpy.around(distribution.sample(4), 4))
        [29.4606  6.3357 30.9928 14.8313]
        >>> print(numpy.around(distribution.mom(1), 4))
        18.5664
    """

    def __init__(self, shape=0.5, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=wrapped_cauchy(shape)*scale, right=shift)
