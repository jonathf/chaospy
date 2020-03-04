"""Chi distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class chi(Dist):
    """Chi distribution."""

    def __init__(self, df=1):
        Dist.__init__(self, df=df)

    def _pdf(self, x, df):
        return x**(df-1.)*numpy.exp(-x*x*0.5)/(2.0)**(df*0.5-1)\
                /special.gamma(df*0.5)

    def _cdf(self, x, df):
        return special.gammainc(df*0.5,0.5*x*x)

    def _ppf(self, q, df):
        return numpy.sqrt(2*special.gammaincinv(df*0.5, q))

    def _lower(self, df):
        return 0.

    def _mom(self, k, df):
        return 2**(.5*k)*special.gamma(.5*(df+k))\
                /special.gamma(.5*df)


class Chi(Add):
    """
    Chi distribution.

    Args:
        df (float, Dist):
            Degrees of freedom
        scale (float, Dist):
            Scaling parameter
        shift (float, Dist):
            Location parameter

    Examples:
        >>> distribution = chaospy.Chi(2, 4, 1)
        >>> distribution
        Chi(df=2, scale=4, shift=1)
        >>> q = numpy.linspace(0, 1, 5)
        >>> distribution.inv(q).round(4)
        array([ 1.    ,  4.0341,  5.7096,  7.6604, 28.1446])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.    , 0.1422, 0.1472, 0.1041, 0.    ])
        >>> distribution.sample(4).round(4)
        array([ 6.8244,  2.9773, 10.8003,  5.5892])
        >>> distribution.mom(1).round(4)
        6.0133
    """

    def __init__(self, df=1, scale=1, shift=0):
        self._repr = {"df": df, "scale": scale, "shift": shift}
        Add.__init__(self, left=chi(df)*scale, right=shift)


class Maxwell(Add):
    """
    Maxwell-Boltzmann distribution
    Chi distribution with 3 degrees of freedom

    Args:
        scale (float, Dist):
            Scaling parameter
        shift (float, Dist):
            Location parameter

    Examples:
        >>> distribution = chaospy.Maxwell(2, 3)
        >>> distribution
        Maxwell(scale=2, shift=3)
        >>> q = numpy.linspace(0, 1, 5)
        >>> distribution.inv(q).round(4)
        array([ 3.    ,  5.2023,  6.0763,  7.0538, 17.0772])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.    , 0.2638, 0.2892, 0.2101, 0.    ])
        >>> distribution.sample(4).round(4)
        array([6.6381, 4.6119, 8.5955, 6.015 ])
        >>> distribution.mom(1).round(4)
        6.1915
    """

    def __init__(self, scale=1, shift=0):
        self._repr = {"scale": scale, "shift": shift}
        Add.__init__(self, left=chi(3)*scale, right=shift)


class Rayleigh(Add):
    """
    Rayleigh distribution

    Args:
        scale (float, Dist):
            Scaling parameter
        shift (float, Dist):
            Location parameter

    Examples:
        >>> distribution = chaospy.Rayleigh(2, 3)
        >>> distribution
        Rayleigh(scale=2, shift=3)
        >>> q = numpy.linspace(0, 1, 5)
        >>> distribution.inv(q).round(4)
        array([ 3.    ,  4.5171,  5.3548,  6.3302, 16.5723])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.    , 0.2844, 0.2944, 0.2081, 0.    ])
        >>> distribution.sample(4).round(4)
        array([5.9122, 3.9886, 7.9001, 5.2946])
        >>> distribution.mom(1).round(4)
        5.5066
    """
    def __init__(self, scale=1, shift=0):
        self._repr = {"scale": scale, "shift": shift}
        Add.__init__(self, left=chi(2)*scale, right=shift)
