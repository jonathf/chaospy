"""
First order statistics functions.
"""
from itertools import product
import numpy

from .. import distributions, poly as polynomials, quad as quadrature


def E(poly, dist=None, **kws):
    """
    Expected value operator.

    1st order statistics of a probability distribution or polynomial on a given
    probability space.

    Args:
        poly (Poly, Dist) : Input to take expected value on.
        dist (Dist) : Defines the space the expected value is taken on.
                It is ignored if `poly` is a distribution.
        **kws (optional) : Extra keywords passed to dist.mom.

    Returns:
        (ndarray) : The expected value of the polynomial or distribution, where
                `expected.shape==poly.shape`.

    Examples:
        >>> x = chaospy.variable()
        >>> Z = chaospy.Uniform()
        >>> print(chaospy.E(Z))
        0.5
        >>> print(chaospy.E(x**3, Z))
        0.25
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


def E_cond(poly, freeze, dist, **kws):

    assert not dist.dependent()

    if poly.dim < len(dist):
        poly = polynomials.setdim(poly, len(dist))

    freeze = polynomials.Poly(freeze)
    freeze = polynomials.setdim(freeze, len(dist))
    keys = freeze.keys
    if len(keys)==1 and keys[0]==(0,)*len(dist):
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
