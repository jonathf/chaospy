"""Conditional expected value."""
from itertools import product

import numpy
import numpoly

from .. import distributions, poly as polynomials, quadrature


def E_cond(poly, freeze, dist, **kws):
    """
    Conditional expected value operator.

    1st order statistics of a polynomial on a given probability space
    conditioned on some of the variables.

    Args:
        poly (Poly):
            Polynomial to find conditional expected value on.
        freeze (numpy.ndarray):
            Boolean values defining the conditional variables. True values
            implies that the value is conditioned on, e.g. frozen during the
            expected value calculation.
        dist (Dist) :
            The distributions of the input used in ``poly``.

    Returns:
        (chaospy.poly.base.Poly) :
            Same as ``poly``, but with the variables not tagged in ``frozen``
            integrated away.

    Examples:
        >>> x, y = chaospy.variable(2)
        >>> poly = chaospy.Poly([1, x, y, 10*x*y])
        >>> dist = chaospy.J(chaospy.Gamma(1, 1), chaospy.Normal(0, 2))
        >>> print(chaospy.E_cond(poly, [1, 0], dist))
        [1.0, q0, 0.0, 0.0]
        >>> print(chaospy.E_cond(poly, [0, 1], dist))
        [1.0, 1.0, q1, 10.0q1]
        >>> print(chaospy.E_cond(poly, [1, 1], dist))
        [1.0, q0, q1, 10.0q0q1]
        >>> print(chaospy.E_cond(poly, [0, 0], dist))
        [1.0, 1.0, 0.0, 0.0]
    """
    poly = polynomials.setdim(poly, len(dist))
    if not poly.isconstant:
        return poly.tonumpy()
    assert not distributions.evaluation.get_dependencies(*dist), dist

    freeze = polynomials.Poly(freeze)
    if freeze.isconstant():
        freeze = freeze.tonumpy().astype(bool)
    else:
        poly, freeze = numpoly.align_exponents(poly, freeze)
        freeze = numpy.isin(poly.keys, freeze.keys)

    poly = polynomials.decompose(poly)

    cache = {}
    if len(freeze.shape) == 1:
        out = _E_cond(poly, freeze, dist, cache, **kws)
    else:
        out = polynomials.concatenate([
            _E_cond(poly, freeze_, dist, cache, **kws)[numpy.newaxis]
            for freeze_ in freeze
        ])
    if out.isconstant():
        out = out.tonumpy()
    return out


def _E_cond(poly, freeze, dist, cache, **kws):
    """Backend for conditional expected value."""
    assert len(poly.names) == len(freeze) == len(dist)
    if numpy.all(freeze):
        return poly.copy()
    poly1 = poly(**{str(var): (var if keep else 1)
                    for var, keep in zip(poly.indeterminants, freeze)})
    poly2 = poly(**{str(var): (var if not keep else 1)
                    for var, keep in zip(poly.indeterminants, freeze)})
    assert len(poly.names) == len(poly2.names)
    # reset non-zero coefficients to 1 so not to duplicate them:
    for key in poly2.keys:
        poly2[poly2 != 0] = 1

    out = numpoly.sum([
        distributions.evaluation.evaluate_moment(
            dist, (exponent*~freeze), cache, **kws)*coefficient
        for exponent, coefficient in zip(poly2.exponents, poly2.coefficients)
    ], axis=0)*poly1

    out, _ = numpoly.align_indeterminants(out, poly)
    return out
