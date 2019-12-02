"""Expected value."""
import numpy

from .. import poly as polynomials


def E(poly, dist=None, **kws):
    """
    Expected value operator.

    1st order statistics of a probability distribution or polynomial on a given
    probability space.

    Args:
        poly (chaospy.poly.ndpoly, Dist):
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
        >>> chaospy.E(dist)
        array([1., 0.])
        >>> x, y = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, x, y, 10*x*y])
        >>> chaospy.E(poly, dist)
        array([1., 1., 0., 0.])
    """
    if dist is None:
        dist, poly = poly, polynomials.variable(len(poly))

    poly = polynomials.setdim(poly, len(dist))
    if not poly.isconstant:
        return poly.tonumpy()

    moments = dist.mom(poly.exponents.T, **kws)
    if len(dist) == 1:
        moments = moments[0]

    out = numpy.zeros(poly.shape)
    for idx, key in enumerate(poly.keys):
        out += poly[key]*moments[idx]
    return out
