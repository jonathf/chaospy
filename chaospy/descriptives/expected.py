"""Expected value."""
import numpy
import numpoly

from .. import distributions, quadrature


def E(poly, dist=None, **kws):
    """
    Expected value operator.

    1st order statistics of a probability distribution or polynomial on a given
    probability space.

    Args:
        poly (numpoly.ndpoly, Dist):
            Input to take expected value on.
        dist (Dist):
            Defines the space the expected value is taken on. It is ignored if
            ``poly`` is a distribution.

    Returns:
        (numpy.ndarray):
            The expected value of the polynomial or distribution, where
            ``expected.shape == poly.shape``.

    Examples:
        >>> dist = chaospy.J(chaospy.Gamma(1, 1), chaospy.Normal(0, 2))
        >>> x, y = numpoly.symbols("x y")
        >>> print(chaospy.E(dist))
        [1. 0.]
        >>> print(chaospy.E(x**2, chaospy.Uniform(0, 3)))
        3.0
        >>> poly = numpoly.polynomial([1, 10*x, y, x*y**2])
        >>> print(chaospy.E(poly, dist))
        [ 1. 10.  0.  4.]
    """
    if isinstance(poly, distributions.Dist):
        dist, poly = poly, numpoly.symbols("q:%d" % len(poly))
    poly = numpoly.polynomial(poly)

    poly, _ = numpoly.align_indeterminants(poly, numpoly.symbols(
        poly.names+tuple("zzz%d__" % idx
                         for idx in range(len(dist)-len(poly.names)))))

    moments = dist.mom(poly.exponents.T, **kws).flatten()
    return sum(coeff*mom for coeff, mom in zip(poly.coefficients, moments))
