"""Conditional expected value."""
from itertools import product

import numpy
import numpoly

from . import expected


def E_cond(poly, freeze, dist, **kws):
    """
    Conditional expected value operator.

    1st order statistics of a polynomial on a given probability space
    conditioned on some of the variables.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to find conditional expected value on.
        freeze (numpy.ndarray):
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
        >>> chaospy.E_cond(poly, [1, 0], dist)
        polynomial([1.0, q0, 0.0, -1.0])
        >>> chaospy.E_cond(poly, [0, 1], dist)
        polynomial([1.0, 1.0, q1, 10.0*q1-1.0])
        >>> chaospy.E_cond(poly, [1, 1], dist)
        polynomial([1, q0, q1, 10*q0*q1-1])
        >>> chaospy.E_cond(poly, [0, 0], dist)
        polynomial([1.0, 1.0, 0.0, -1.0])

    """
    poly = numpoly.set_dimensions(poly, len(dist))
    if not poly.isconstant:
        return poly.tonumpy()
    assert not dist.stochastic_dependent, dist

    freeze = numpoly.polynomial(freeze)
    if freeze.isconstant():
        freeze = freeze.tonumpy().astype(bool)
    else:
        poly, freeze = numpoly.align_exponents(poly, freeze)
        freeze = numpy.isin(poly.keys, freeze.keys)

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
        unfrozen[key] = unfrozen[key] != 0

    return numpoly.sum(frozen*expected.E(unfrozen, dist), 0)
