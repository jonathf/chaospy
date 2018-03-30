"""Beta distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class beta_(Dist):

    def __init__(self, a=1, b=1):
        Dist.__init__(self, a=a, b=b)

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

    def _bnd(self, a, b):
        return 0., 1.


class Beta(Add):
    R"""
    Beta Probability Distribution.

    Args:
        alpha (float, Dist): First shape parameter, alpha > 0
        beta (float, Dist): Second shape parameter, b > 0
        lower (float, Dist): Lower threshold
        upper (float, Dist): Upper threshold

    Examples:
        >>> f = chaospy.Beta(2, 2, 2, 3)
        >>> print(f)
        Beta(alpha=2, beta=2, lower=2, upper=3)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(f.inv(q), 4))
        [2.2871 2.4329 2.5671 2.7129]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(f.pdf(f.inv(q)), 4))
        [1.2281 1.473  1.473  1.2281]
        >>> print(numpy.around(f.sample(4), 4))
        [2.6039 2.2112 2.8651 2.4881]
        >>> print(f.mom(1))
        2.5
    """

    def __init__(self, alpha, beta, lower=0, upper=1):
        self._repr = {
            "alpha": alpha, "beta": beta, "lower": lower, "upper": upper}
        Add.__init__(self, left=beta_(alpha, beta)*(upper-lower), right=lower)


class ArcSinus(Add):
    """
    Generalized Arc-sinus distribution

    Args:
        shape (float, Dist): Shape parameter where 0.5 is the default
            non-generalized case. Defined on the interval ``[0, 1]``.
        lower (float, Dist): Lower threshold
        upper (float, Dist): Upper threshold

    Examples:
        >>> distribution = chaospy.ArcSinus(0.8, 4, 6)
        >>> print(distribution)
        ArcSinus(lower=4, shape=0.8, upper=6)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [4.9875 5.6438 5.9134 5.9885 5.9996]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.1667 0.3333 0.5    0.6667 0.8333]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [ 0.1857  0.3868  1.1633  5.8145 92.8592]
        >>> print(numpy.around(distribution.sample(4), 4))
        [5.9861 4.6882 6.     5.897 ]
        >>> print(distribution.mom(1))
        5.6
    """

    def __init__(self, shape=0.5, lower=0, upper=1):
        self._repr = {"shape": shape, "lower": lower, "upper": upper}
        Add.__init__(
            self, left=beta_(shape, 1-shape)*(upper-lower), right=lower)


class PowerLaw(Add):
    """
    Powerlaw distribution

    Args:
        shape (float, Dist) : Shape parameter
        lower (float, Dist) : Location of lower threshold
        upper (float, Dist) : Location of upper threshold

    Examples:
        >>> distribution = chaospy.PowerLaw(0.8, 4, 6)
        >>> print(distribution)
        PowerLaw(lower=4, shape=0.8, upper=6)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [4.213  4.5066 4.8409 5.2048 5.5924]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.1667 0.3333 0.5    0.6667 0.8333]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.626  0.5264 0.4757 0.4427 0.4187]
        >>> print(numpy.around(distribution.sample(4), 4))
        [5.1753 4.1339 5.8765 4.8036]
        >>> print(numpy.around(distribution.mom(1), 4))
        4.8889
    """

    def __init__(self, shape=1, lower=0, upper=1):
        self._repr = {"shape": shape, "lower": lower, "upper": upper}
        Add.__init__(self, left=beta_(shape, 1)*(upper-lower), right=lower)


class Wigner(Add):
    """
    Wigner (semi-circle) distribution

    Args:
        radius (float, Dist) : radius of the semi-circle (scale)
        shift (float, Dist) : location of the circle origin (location)

    Examples:
        >>> distribution = chaospy.Wigner(2, 3)
        >>> print(distribution)
        Wigner(radius=2, shift=3)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [1.8934 2.4701 3.     3.5299 4.1066]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.1667 0.3333 0.5    0.6667 0.8333]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.2651 0.3069 0.3183 0.3069 0.2651]
        >>> print(numpy.around(distribution.sample(4), 4))
        [3.4874 1.6895 4.6123 2.944 ]
        >>> print(numpy.around(distribution.mom(1), 4))
        3.0
    """

    def __init__(self, radius=1, shift=0):
        self._repr = {"radius": radius, "shift": shift}
        Add.__init__(self, left=radius*(2*beta_(1.5, 1.5)-1), right=shift)
