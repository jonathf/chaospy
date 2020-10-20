"""Beta distribution."""
import numpy
from scipy import special
import chaospy

from ..baseclass import SimpleDistribution, LowerUpperDistribution, ShiftScaleDistribution


class beta_(SimpleDistribution):

    def __init__(self, a=1, b=1):
        super(beta_, self).__init__(dict(a=a, b=b))

    def _pdf(self, x, a, b):
        out = x**(a-1)*(1-x)**(b-1)/special.beta(a, b)
        out = numpy.where(numpy.isfinite(out), out, 0)
        return out

    def _cdf(self, x, a, b):
        return special.btdtr(a, b, x)

    def _ppf(self, qloc, a, b):
        return special.btdtri(a, b, qloc)

    def _mom(self, k, a, b):
        return special.beta(a+k,b)/special.beta(a,b)

    def _ttr(self, n, a, b):
        nab = 2*n+a+b
        A = ((a-1)**2-(b-1)**2)*.5/\
                (nab*(nab-2) + (nab==0) + (nab==2)) + .5
        B1 = a*b*1./((a+b+1)*(a+b)**2)
        B2 = (n+a-1)*(n+b-1)*n*(n+a+b-2.)/\
            ((nab-1)*(nab-3)*(nab-2)**2+2.*((n==0)+(n==1)))
        B = numpy.where((n==0)+(n==1), B1, B2)
        return A, B

    def _lower(self, a, b):
        return 0.

    def _upper(self, a, b):
        return 1.


class Beta(LowerUpperDistribution):
    R"""
    Beta Probability Distribution.

    Args:
        alpha (float, Distribution):
            First shape parameter, alpha > 0
        beta (float, Distribution):
            Second shape parameter, b > 0
        lower (float, Distribution):
            Lower threshold
        upper (float, Distribution):
            Upper threshold

    Examples:
        >>> distribution = chaospy.Beta(1.5, 3.5)
        >>> distribution
        Beta(1.5, 3.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.126, 0.222, 0.326, 0.464, 1.   ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 2.066, 2.051, 1.734, 1.168, 0.   ])
        >>> distribution.sample(4).round(3)
        array([0.358, 0.083, 0.651, 0.263])
        >>> distribution.mom(1).round(4)
        0.3
        >>> distribution.ttr([0, 1, 2, 3]).round(4)
        array([[0.3   , 0.4143, 0.4524, 0.4697],
               [0.035 , 0.035 , 0.0478, 0.0535]])

    """

    def __init__(self, alpha, beta, lower=0, upper=1):
        super(Beta, self).__init__(
            dist=beta_(alpha, beta),
            lower=lower,
            upper=upper,
            repr_args=[alpha, beta],
        )


class ArcSinus(LowerUpperDistribution):
    """
    Generalized Arc-sinus distribution

    Args:
        shape (float, Distribution):
            Shape parameter where 0.5 is the default non-generalized case.
            Defined on the interval ``[0, 1]``.
        lower (float, Distribution):
            Lower threshold
        upper (float, Distribution):
            Upper threshold

    Examples:
        >>> distribution = chaospy.ArcSinus(0.5)
        >>> distribution
        ArcSinus(0.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.095, 0.345, 0.655, 0.905, 1.   ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 1.083, 0.669, 0.669, 1.083, 0.   ])
        >>> distribution.sample(4).round(3)
        array([0.732, 0.032, 0.994, 0.472])
        >>> distribution.mom(1).round(4)
        0.5
        >>> distribution.ttr([0, 1, 2, 3]).round(4)
        array([[0.5   , 0.5   , 0.5   , 0.5   ],
               [0.125 , 0.125 , 0.0625, 0.0625]])

    """

    def __init__(self, shape=0.5, lower=0, upper=1):
        super(ArcSinus, self).__init__(
            dist=beta_(shape, 1-shape),
            lower=lower,
            upper=upper,
            repr_args=[shape],
        )


class PowerLaw(LowerUpperDistribution):
    """
    Powerlaw distribution

    Args:
        shape (float, Distribution):
            Shape parameter
        lower (float, Distribution):
            Location of lower threshold
        upper (float, Distribution):
            Location of upper threshold

    Examples:
        >>> distribution = chaospy.PowerLaw(0.8)
        >>> distribution
        PowerLaw(0.8)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.134, 0.318, 0.528, 0.757, 1.   ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 1.196, 1.006, 0.909, 0.846, 0.8  ])
        >>> distribution.sample(4).round(3)
        array([0.588, 0.067, 0.938, 0.402])
        >>> distribution.mom(1).round(4)
        0.4444
        >>> distribution.ttr([0, 1, 2, 3]).round(4)
        array([[0.4444, 0.5029, 0.5009, 0.5004],
               [0.0882, 0.0882, 0.0668, 0.0643]])

    """

    def __init__(self, shape=1, lower=0, upper=1):
        super(PowerLaw, self).__init__(
            dist=beta_(shape, 1),
            lower=lower,
            upper=upper,
            repr_args=[shape],
        )


class Wigner(ShiftScaleDistribution):
    """
    Wigner (semi-circle) distribution

    Args:
        radius (float, Distribution):
            Radius of the semi-circle (scale)
        shift (float, Distribution):
            Location of the circle origin (location)

    Examples:
        >>> distribution = chaospy.Wigner(1.5)
        >>> distribution
        Wigner(1.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([-1.5  , -0.738, -0.237,  0.237,  0.738,  1.5  ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.37 , 0.419, 0.419, 0.37 , 0.   ])
        >>> distribution.sample(4).round(3)
        array([ 0.366, -0.983,  1.209, -0.042])
        >>> distribution.mom(1).round(4)
        0.0
        >>> distribution.ttr([0, 1, 2, 3]).round(4)
        array([[0.    , 0.    , 0.    , 0.    ],
               [0.5625, 0.5625, 0.5625, 0.5625]])

    """

    def __init__(self, radius=1, shift=0):
        super(Wigner, self).__init__(
            dist=beta_(1.5, 1.5),
            scale=2*radius, shift=shift-radius)
        self._repr_args = [radius]+chaospy.format_repr_kwargs(shift=(shift, 0))


class PERT(Beta):
    r"""
    Program Evaluation and Review Technique (PERT) Distribution.

    Defined by its mean::

        \mu = \frac{lower + gamma*mode + upper}{2 + gamma}

    Normal PERT for `gamma=4`. Other values results in the so called
    modified-PERT distribution.

    Args:
        lower (float):
            The lower bounds for the distribution.
        mode (float, Distribution):
            The mode of the distribution.
        upper (float):
            The upper bounds for the distribution.
        gamma (flat, Distribution):
            Modify the PERT distribution to make more emphasis on the
            distribution mode instead of the distribution tails.

    Examples:
        >>> distribution = chaospy.PERT(-1, 0, 1)
        >>> distribution
        PERT(-1, 0, 1)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([-1.   , -0.347, -0.107,  0.107,  0.347,  1.   ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.726, 0.916, 0.916, 0.726, 0.   ])
        >>> distribution.sample(4).round(3)
        array([ 0.167, -0.479,  0.622, -0.019])
        >>> distribution.mom(1).round(4)
        0.0
        >>> distribution.ttr([0, 1, 2, 3]).round(4)
        array([[0.    , 0.    , 0.    , 0.    ],
               [0.1429, 0.1429, 0.1905, 0.2121]])

    """

    def __init__(self, lower, mode, upper, gamma=4):
        mu = (lower+4*mode+upper)/6.
        alpha = 1+gamma*(mu-lower)/(upper-lower)
        beta = 1+gamma*(upper-mu)/(upper-lower)
        LowerUpperDistribution.__init__(
            self,
            dist=beta_(alpha, beta),
            lower=lower,
            upper=upper,
        )
        self._repr_args = [lower, mode, upper]
        self._repr_args += chaospy.format_repr_kwargs(gamma=(gamma, 4))
