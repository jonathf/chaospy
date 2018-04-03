"""Burr Type XII or Singh-Maddala distribution."""
from ..baseclass import Dist
from ..operators.addition import Add


class burr(Dist):
    """Stadard Burr distribution."""

    def __init__(self, alpha=1., kappa=1.):
        Dist.__init__(self, alpha=alpha, kappa=kappa)

    def _pdf(self, x, alpha, kappa):
        output = alpha*kappa*x**(-alpha-1.0)
        output /= (1+x**alpha)**(kappa+1)
        return output

    def _cdf(self, x, alpha, kappa):
        return 1-(1+x**alpha)**-kappa

    def _ppf(self, q, alpha, kappa):
        return ((1-q)**(-1./kappa) -1)**(1./alpha)

    def _bnd(self, x, alpha, kappa):
        return 0, self._ppf(1-1e-10, alpha, kappa)


class Burr(Add):
    """
    Burr Type XII or Singh-Maddala distribution.

    Args:
        alpha (float, Dist): First shape parameter
        kappa (float, Dist): Second shape parameter
        loc (float, Dist): Location parameter
        scale (float, Dist): Scaling parameter

    Examples:
        >>> distribution = chaospy.Burr(1.2, 1.2, 4, 2)
        >>> print(distribution)
        Burr(alpha=1.2, kappa=1.2, loc=4, scale=2)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [4.4435 4.9358 5.6291 6.8009 9.6146]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.1667 0.3333 0.5    0.6667 0.8333]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [1.41648e+01 1.82020e+00 3.17300e-01 4.58000e-02 2.80000e-03]
        >>> print(numpy.around(distribution.sample(4), 4))
        [ 6.6775  4.311  18.9718  5.5396]
    """

    def __init__(self, alpha=1, kappa=1, loc=0, scale=1):
        self._repr = {
            "alpha": alpha, "kappa": kappa, "loc": loc, "scale": scale}
        Add.__init__(self, left=burr(alpha, kappa)*scale, right=loc)
