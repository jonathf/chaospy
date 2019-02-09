"""Expected value."""
import numpy

from .. import distributions, poly as polynomials, quad as quadrature


def E(poly, dist=None, **kws):
    """
    Expected value operator.

    1st order statistics of a probability distribution or polynomial on a given
    probability space.

    Args:
        poly (Poly, Dist):
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
        >>> print(chaospy.E(dist))
        [1. 0.]
        >>> x, y = chaospy.variable(2)
        >>> poly = chaospy.Poly([1, x, y, 10*x*y])
        >>> print(chaospy.E(poly, dist))
        [1. 1. 0. 0.]
    """
    if not isinstance(poly, (distributions.Dist, polynomials.Poly)):
        print(type(poly))
        print("Approximating expected value...")
        out = quadrature.quad(poly, dist, veceval=True, **kws)
        print("done")
        return out

    if isinstance(poly, distributions.Dist):
        dist, poly = poly, polynomials.variable(len(poly))

    if not poly.keys:
        return numpy.zeros(poly.shape, dtype=int)

    if isinstance(poly, (list, tuple, numpy.ndarray)):
        return [E(_, dist, **kws) for _ in poly]

    if poly.dim < len(dist):
        poly = polynomials.setdim(poly, len(dist))

    shape = poly.shape
    poly = polynomials.flatten(poly)

    keys = poly.keys
    mom = dist.mom(numpy.array(keys).T, **kws)
    A = poly.A

    if len(dist) == 1:
        mom = mom[0]

    out = numpy.zeros(poly.shape)
    for i in range(len(keys)):
        out += A[keys[i]]*mom[i]

    out = numpy.reshape(out, shape)
    return out
