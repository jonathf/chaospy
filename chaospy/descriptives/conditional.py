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
        [1. 1. 0. 0.]
        >>> print(chaospy.E_cond(poly, numpy.eye(2), dist))
        [[1.0 x 0.0 0.0]
         [1.0 1.0 y 10.0*y]]
    """
    if isinstance(poly, chaospy.Dist):
        dist, poly = poly, numpoly.symbols("q:%d" % len(poly))
    poly = numpoly.polynomial(poly)
    poly, _ = numpoly.align_indeterminants(poly, numpoly.symbols(
        poly.names+tuple("zzz%d__" % idx
                         for idx in range(len(dist)-len(poly.names)))))

    assert not chaospy.distributions.evaluation.get_dependencies(*dist), dist

    freeze = numpoly.polynomial(freeze)
    if freeze.isconstant():
        freeze = freeze.toarray().astype(bool)
    else:
        poly, freeze = numpoly.align_exponents(poly, freeze)
        freeze = numpy.isin(poly.keys, freeze.keys)

    cache = {}
    if len(freeze.shape) == 1:
        out = _E_cond(poly, freeze, dist, cache, **kws)
    else:
        out = numpoly.concatenate([
            _E_cond(poly, freeze_, dist, cache, **kws)[numpy.newaxis]
            for freeze_ in freeze
        ])
    if out.isconstant():
        out = out.toarray()
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
        chaospy.distributions.evaluation.evaluate_moment(
            dist, (exponent*~freeze), cache, **kws)*coefficient
        for exponent, coefficient in zip(poly2.exponents, poly2.coefficients)
    ], axis=0)*poly1

    out, _ = numpoly.align_indeterminants(out, poly)
    return out
