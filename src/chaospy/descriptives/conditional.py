"""Conditional expected value."""
from itertools import product
import numpy

from .. import distributions, poly as polynomials, quad as quadrature


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
    if poly.dim < len(dist):
        poly = polynomials.setdim(poly, len(dist))

    freeze = polynomials.Poly(freeze)
    freeze = polynomials.setdim(freeze, len(dist))
    keys = freeze.keys
    if len(keys) == 1 and keys[0] == (0,)*len(dist):
        freeze = list(freeze.A.values())[0]
    else:
        freeze = numpy.array(keys)
    freeze = freeze.reshape(int(freeze.size/len(dist)), len(dist))

    shape = poly.shape
    poly = polynomials.flatten(poly)

    kmax = numpy.max(poly.keys, 0) + 1
    keys = [range(k) for k in kmax]

    A = poly.A.copy()
    keys = poly.keys
    out = {}
    zeros = [0]*poly.dim

    for i in range(len(keys)):

        key = list(keys[i])
        a = A[tuple(key)]

        for d in range(poly.dim):
            for j in range(len(freeze)):
                if freeze[j, d]:
                    key[d], zeros[d] = zeros[d], key[d]
                    break

        tmp = a*dist.mom(tuple(key))
        if tuple(zeros) in out:
            out[tuple(zeros)] = out[tuple(zeros)] + tmp
        else:
            out[tuple(zeros)] = tmp

        for d in range(poly.dim):
            for j in range(len(freeze)):
                if freeze[j, d]:
                    key[d], zeros[d] = zeros[d], key[d]
                    break

    out = polynomials.Poly(out, poly.dim, poly.shape, float)
    out = polynomials.reshape(out, shape)

    return out
