"""Conditional expected value."""
import numpy
import chaospy
import numpoly

from . import expected


def E_cond(poly, freeze, dist, **kws):
    """
    Conditional expected value of a distribution or polynomial.

    1st order statistics of a polynomial on a given probability space
    conditioned on some of the variables.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to find conditional expected value on.
        freeze (numpy.ndpoly):
            Boolean values defining the conditional variables. True values
            implies that the value is conditioned on, e.g. frozen during the
            expected value calculation.
        dist (Distribution) :
            The distributions of the input used in ``poly``.

    Returns:
        (numpoly.ndpoly) :
            Same as ``poly``, but with the variables not tagged in ``frozen``
            integrated away.

    Examples:
        >>> q0, q1 = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, q0, q1, 10*q0*q1-1])
        >>> poly
        polynomial([1, q0, q1, 10*q0*q1-1])
        >>> dist = chaospy.J(chaospy.Gamma(1, 1), chaospy.Normal(0, 2))
        >>> chaospy.E_cond(poly, q0, dist)
        polynomial([1.0, q0, 0.0, -1.0])
        >>> chaospy.E_cond(poly, q1, dist)
        polynomial([1.0, 1.0, q1, 10.0*q1-1.0])
        >>> chaospy.E_cond(poly, [q0, q1], dist)
        polynomial([1, q0, q1, 10*q0*q1-1])
        >>> chaospy.E_cond(poly, [], dist)
        polynomial([1.0, 1.0, 0.0, -1.0])
        >>> chaospy.E_cond(4, [], dist)
        array(4)

    """
    poly = numpoly.set_dimensions(poly, len(dist))
    if poly.isconstant():
        return poly.tonumpy()
    assert not dist.stochastic_dependent, dist

    freeze = numpoly.aspolynomial(freeze)
    if not freeze.size:
        return numpoly.polynomial(chaospy.E(poly, dist))
    if not freeze.isconstant():
        freeze = [name in freeze.names for name in poly.names]
    else:
        freeze = freeze.tonumpy()
    freeze = numpy.asarray(freeze, dtype=bool)

    # decompose into frozen and unfrozen part
    poly = numpoly.decompose(poly)
    unfrozen = poly(**{
        ("q%d" % idx): 1 for idx, keep in enumerate(freeze) if keep})
    frozen = poly(**{
        ("q%d" % idx): 1 for idx, keep in enumerate(freeze) if not keep})

    # if no unfrozen, poly will return numpy.ndarray instead of numpoly.ndpoly
    if not isinstance(unfrozen, numpoly.ndpoly):
        return numpoly.sum(frozen, 0)

    # Remove frozen coefficients, such that poly == sum(frozen*unfrozen) holds
    for key in unfrozen.keys:
        unfrozen.values[key] = unfrozen.values[key] != 0
    return numpoly.sum(frozen*expected.E(unfrozen, dist), 0)
