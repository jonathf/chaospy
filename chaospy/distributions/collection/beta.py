"""Beta distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators import ShiftScale


class beta_(Dist):

    def __init__(self, a=1, b=1):
        super(beta_, self).__init__(a=a, b=b)

    def _pdf(self, x, a, b):
        return x**(a-1)*(1-x)**(b-1)/ \
            special.beta(a, b)

    def _cdf(self, x, a, b):
        return special.btdtr(a, b, x)

    def _ppf(self, q, a, b):
        return special.btdtri(a, b, q)

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


class Beta(ShiftScale):
    R"""
    Beta Probability Distribution.

    Args:
        alpha (float, Dist):
            First shape parameter, alpha > 0
        beta (float, Dist):
            Second shape parameter, b > 0
        lower (float, Dist):
            Lower threshold
        upper (float, Dist):
            Upper threshold

    Examples:
        >>> distribution = chaospy.Beta(2, 2, 2, 3)
        >>> distribution
        Beta(alpha=2, beta=2, lower=2, upper=3)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([2.2871, 2.4329, 2.5671, 2.7129])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([1.2281, 1.473 , 1.473 , 1.2281])
        >>> distribution.sample(4).round(4)
        array([2.6039, 2.2112, 2.8651, 2.4881])
        >>> distribution.mom(1).round(4)
        2.5
        >>> distribution.ttr([1, 2, 3]).round(4)
        array([[2.5   , 2.5   , 2.5   ],
               [0.05  , 0.0571, 0.0595]])
    """

    def __init__(self, alpha, beta, lower=0, upper=1):
        self._repr = {
            "alpha": alpha, "beta": beta, "lower": lower, "upper": upper}
        super(Beta, self).__init__(dist=beta_(alpha, beta), scale=upper-lower, shift=lower)


class ArcSinus(ShiftScale):
    """
    Generalized Arc-sinus distribution

    Args:
        shape (float, Dist):
            Shape parameter where 0.5 is the default non-generalized case.
            Defined on the interval ``[0, 1]``.
        lower (float, Dist):
            Lower threshold
        upper (float, Dist):
            Upper threshold

    Examples:
        >>> distribution = chaospy.ArcSinus(0.8, 4, 6)
        >>> distribution
        ArcSinus(lower=4, shape=0.8, upper=6)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> distribution.inv(q).round(4)
        array([4.9875, 5.6438, 5.9134, 5.9885, 5.9996])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.1667, 0.3333, 0.5   , 0.6667, 0.8333])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([ 0.1857,  0.3868,  1.1633,  5.8145, 92.8592])
        >>> distribution.sample(4).round(4)
        array([5.9861, 4.6882, 6.    , 5.897 ])
        >>> distribution.mom(1)
        array(5.6)
    """

    def __init__(self, shape=0.5, lower=0, upper=1):
        self._repr = {"shape": shape, "lower": lower, "upper": upper}
        super(ArcSinus, self).__init__(dist=beta_(shape, 1-shape), scale=upper-lower, shift=lower)


class PowerLaw(ShiftScale):
    """
    Powerlaw distribution

    Args:
        shape (float, Dist):
            Shape parameter
        lower (float, Dist):
            Location of lower threshold
        upper (float, Dist):
            Location of upper threshold

    Examples:
        >>> distribution = chaospy.PowerLaw(0.8, 4, 6)
        >>> distribution
        PowerLaw(lower=4, shape=0.8, upper=6)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> distribution.inv(q).round(4)
        array([4.213 , 4.5066, 4.8409, 5.2048, 5.5924])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.1667, 0.3333, 0.5   , 0.6667, 0.8333])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.626 , 0.5264, 0.4757, 0.4427, 0.4187])
        >>> distribution.sample(4).round(4)
        array([5.1753, 4.1339, 5.8765, 4.8036])
        >>> distribution.mom(1).round(4)
        4.8889
    """

    def __init__(self, shape=1, lower=0, upper=1):
        self._repr = {"shape": shape, "lower": lower, "upper": upper}
        super(PowerLaw, self).__init__(dist=beta_(shape, 1), scale=(upper-lower), shift=lower)


class Wigner(ShiftScale):
    """
    Wigner (semi-circle) distribution

    Args:
        radius (float, Dist):
            Radius of the semi-circle (scale)
        shift (float, Dist):
            Location of the circle origin (location)

    Examples:
        >>> distribution = chaospy.Wigner(2, 3)
        >>> distribution
        Wigner(radius=2, shift=3)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> distribution.inv(q).round(4)
        array([1.8934, 2.4701, 3.    , 3.5299, 4.1066])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.1667, 0.3333, 0.5   , 0.6667, 0.8333])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.2651, 0.3069, 0.3183, 0.3069, 0.2651])
        >>> distribution.sample(4).round(4)
        array([3.4874, 1.6895, 4.6123, 2.944 ])
        >>> distribution.mom(1).round(4)
        3.0
    """

    def __init__(self, radius=1, shift=0):
        self._repr = {"radius": radius, "shift": shift}
        super(Wigner, self).__init__(
            dist=beta_(1.5, 1.5), scale=2*radius, shift=shift-radius)


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
        mode (float, Dist):
            The mode of the distribution.
        upper (float):
            The upper bounds for the distribution.
        gamma (flat, Dist):
            Modify the PERT distribution to make more emphasis on the
            distribution mode instead of the distribution tails.

    Examples:
        >>> distribution = chaospy.PERT(-1, 0, 1)
        >>> distribution
        PERT(gamma=4, lower=-1, mode=0, upper=1)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> distribution.inv(q).round(4)
        array([-0.3946, -0.1817,  0.    ,  0.1817,  0.3946])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.1667, 0.3333, 0.5   , 0.6667, 0.8333])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.6683, 0.8766, 0.9375, 0.8766, 0.6683])
        >>> distribution.sample(4).round(4)
        array([ 0.1669, -0.4788,  0.6223, -0.019 ])
        >>> distribution.mom(1).round(4)
        0.0
    """

    def __init__(self, lower, mode, upper, gamma=4):
        mu = (lower+4*mode+upper)/6.
        alpha = 1+gamma*(mu-lower)/(upper-lower)
        beta = 1+gamma*(upper-mu)/(upper-lower)
        super(PERT, self).__init__(alpha=alpha, beta=beta, lower=lower, upper=upper)
        self._repr = {"lower": lower, "mode": mode, "upper": upper, "gamma": gamma}
