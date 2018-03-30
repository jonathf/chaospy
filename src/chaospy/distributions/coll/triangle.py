"""Triangle probability distribution."""
import numpy
from scipy import special, misc

from ..baseclass import Dist
from ..operators.addition import Add
from .beta import beta_


def tri_ttr(k, a):
    """
    Custom TTR function.

    Triangle distribution does not have an analytical TTR function, but because
    of its non-smooth nature, a blind integration scheme will converge very
    slowly. However, by splitting the integration into two divided at the
    discontinuity in the derivative, TTR can be made operative.
    """
    from chaospy.quadrature import clenshaw_curtis
    q1,w1 = clenshaw_curtis(int(10**3*a), 0, a)
    q2,w2 = clenshaw_curtis(int(10**3*(1-a)), a, 1)
    q = numpy.concatenate([q1,q2], 1)
    w = numpy.concatenate([w1,w2])
    w = w*numpy.where(q<a, 2*q/a, 2*(1-q)/(1-a))

    from chaospy.poly import variable
    x = variable()

    orth = [x*0, x**0]
    inner = numpy.sum(q*w, -1)
    norms = [1., 1.]
    A,B = [],[]

    for n in range(k):
        A.append(inner/norms[-1])
        B.append(norms[-1]/norms[-2])
        orth.append((x-A[-1])*orth[-1]-orth[-2]*B[-1])

        y = orth[-1](*q)**2*w
        inner = numpy.sum(q*y, -1)
        norms.append(numpy.sum(y, -1))

    A, B = numpy.array(A).T[0], numpy.array(B).T
    return A, B


class triangle(Dist):
    """Triangle probability distribution."""

    def __init__(self, a=.5):
        assert numpy.all(a>=0) and numpy.all(a<=1)
        Dist.__init__(self, a=a)

    def _pdf(self, D, a):
        return numpy.where(D<a, 2*D/a, 2*(1-D)/(1-a))

    def _cdf(self, D, a):
        return numpy.where(D<a, D**2/(a + (a==0)),
                (2*D-D*D-a)/(1-a+(a==1)))

    def _ppf(self, q, a):
        return numpy.where(q<a, numpy.sqrt(q*a), 1-numpy.sqrt(1-a-q*(1-a)))

    def _mom(self, k, a):
        a_ = a*(a!=1)
        out = 2*(1.-a_**(k+1))/((k+1)*(k+2)*(1-a_))
        return numpy.where(a==1, 2./(k+2), out)

    def _bnd(self, a):
        return 0., 1.

    def _ttr(self, k, a):
        a = a.item()
        if a==0: return beta_()._ttr(k, 1, 2)
        if a==1: return beta_()._ttr(k, 2, 1)

        A,B = tri_ttr(numpy.max(k)+1, a)
        A = numpy.array([[A[_] for _ in k[0]]])
        B = numpy.array([[B[_] for _ in k[0]]])
        return A,B


class Triangle(Add):
    """
    Triangle Distribution.

    Must have lower <= midpoint <= upper.

    Args:
        lower (float, Dist) : Lower bound
        midpoint (float, Dist) : Location of the top
        upper (float, Dist) : Upper bound

    Examples:
        >>> f = chaospy.Triangle(2, 3, 4)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(f.inv(q), 4))
        [2.6325 2.8944 3.1056 3.3675]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(f.pdf(f.inv(q)), 4))
        [0.6325 0.8944 0.8944 0.6325]
        >>> print(numpy.around(f.sample(4), 4))
        [3.1676 2.4796 3.6847 2.982 ]
        >>> print(numpy.around(f.mom(1), 4))
        3.0
    """

    def __init__(self, lower, midpoint, upper):
        self._repr = {"lower": lower, "midpoint": midpoint, "upper": upper}
        midpoint = (midpoint-lower)*1./(upper-lower)
        Add.__init__(self, left=triangle(midpoint)*(upper-lower), right=lower)
