"""Conditional expected value."""
import numpy
import numpoly
import chaospy


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
        (numpoly.ndpoly) :
            Same as ``poly``, but with the variables not tagged in ``frozen``
            integrated away.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> poly = numpoly.polynomial([1, x, y, 10*x*y])
        >>> dist = chaospy.J(chaospy.Gamma(1, 1), chaospy.Normal(0, 2))
        >>> print(chaospy.E_cond(poly, [1, 0], dist))
        [1.0 x 0.0 0.0]
        >>> print(chaospy.E_cond(poly, y, dist))
        [1 x y 10*x*y]
        >>> print(chaospy.E_cond(poly, [x, y], dist))
        [1 x y 10*x*y]
        >>> print(chaospy.E_cond(poly, [0, 0], dist))
        [1.0 1.0 0.0 0.0]
    """
    freeze = numpoly.polynomial(freeze)
    if freeze.isconstant():
        freeze = freeze.astype(bool)
    else:
        poly, freeze = numpoly.align_exponents(poly, freeze)
        freeze = numpy.isin(poly.keys, freeze.keys)
    if numpy.all(freeze):
        return poly.copy()

    poly_indep, poly_dep = _split_out_conditions(poly, freeze)
    dist = [dist_ for dist_, keep in zip(dist, freeze) if not keep]

    out = chaospy.E(poly_dep, chaospy.J(*dist))*poly_indep
    out, _ = numpoly.align_indeterminants(out, poly)
    return out


def _split_out_conditions(poly, freeze):
    poly1 = poly(**{str(var): (var if keep else 1)
                    for var, keep in zip(poly.indeterminants, freeze)})

    poly2 = poly(**{str(var): (var if not keep else 1)
                    for var, keep in zip(poly.indeterminants, freeze)})
    # reset non-zero coefficients to 1 to not duplciating them:
    for key in poly2.keys:
        poly2[key][numpy.nonzero(poly2[key])] = 1
    return poly1, poly2
