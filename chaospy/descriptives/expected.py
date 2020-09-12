"""Expected value."""
import numpy
import numpoly


def E(poly, dist=None, **kws):
    """
    Expected value operator.

    1st order statistics of a probability distribution or polynomial on a given
    probability space.

    Args:
        poly (numpoly.ndpoly, Distribution):
            Input to take expected value on.
        dist (Distribution):
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
        >>> q0, q1 = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, q0, q1, 10*q0*q1-1])
        >>> chaospy.E(poly, dist)
        array([ 1.,  1.,  0., -1.])

    """
    if dist is None:
        dist, poly = poly, numpoly.variable(len(poly))

    poly = numpoly.set_dimensions(poly, len(dist))
    if poly.isconstant():
        return poly.tonumpy()

    moments = dist.mom(poly.exponents.T, **kws)
    if len(dist) == 1:
        moments = moments[0]

    out = numpy.zeros(poly.shape)
    for idx, key in enumerate(poly.keys):
        out += poly[key]*moments[idx]
    return out
