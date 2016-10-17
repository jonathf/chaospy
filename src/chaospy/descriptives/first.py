"""
First order statistics functions.
"""
from itertools import product
import numpy as np

import chaospy as cp


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
        >>> x = cp.variable()
        >>> Z = cp.Uniform()
        >>> print(cp.E(Z))
        0.5
        >>> print(cp.E(x**3, Z))
        0.25
    """
    if not isinstance(poly, (cp.dist.Dist, cp.poly.Poly)):
        print(type(poly))
        print("Approximating expected value...")
        out = cp.quadrature.quad(poly, dist, veceval=True, **kws)
        print("done")
        return out

    if isinstance(poly, cp.dist.Dist):
        dist = poly
        poly = cp.poly.variable(len(poly))

    if not poly.keys:
        return np.zeros(poly.shape, dtype=int)

    if isinstance(poly, (list, tuple, np.ndarray)):
        return [E(_, dist, **kws) for _ in poly]

    if poly.dim < len(dist):
        poly = cp.poly.setdim(poly, len(dist))

    shape = poly.shape
    poly = cp.poly.flatten(poly)

    keys = poly.keys
    mom = dist.mom(np.array(keys).T, **kws)
    A = poly.A

    if len(dist)==1:
        mom = mom[0]

    out = np.zeros(poly.shape)
    for i in range(len(keys)):
        out += A[keys[i]]*mom[i]

    out = np.reshape(out, shape)
    return out


def E_cond(poly, freeze, dist, **kws):

    assert not dist.dependent()

    if poly.dim < len(dist):
        poly = cp.poly.setdim(poly, len(dist))

    freeze = cp.poly.Poly(freeze)
    freeze = cp.poly.setdim(freeze, len(dist))
    keys = freeze.keys
    if len(keys)==1 and keys[0]==(0,)*len(dist):
        freeze = list(freeze.A.values())[0]
    else:
        freeze = np.array(keys)
    freeze = freeze.reshape(int(freeze.size/len(dist)), len(dist))

    shape = poly.shape
    poly = cp.poly.flatten(poly)

    kmax = np.max(poly.keys, 0) + 1
    keys = [range(k) for k in kmax]
    keys = [k for k in product(*keys)]

    vals = dist.mom(np.array(keys).T, **kws).T
    mom = dict(zip(keys, vals))

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

        tmp = a*mom[tuple(key)]
        if tuple(zeros) in out:
            out[tuple(zeros)] = out[tuple(zeros)] + tmp
        else:
            out[tuple(zeros)] = tmp

        for d in range(poly.dim):
            for j in range(len(freeze)):
                if freeze[j, d]:
                    key[d], zeros[d] = zeros[d], key[d]
                    break

    out = cp.poly.Poly(out, poly.dim, poly.shape, float)
    out = cp.poly.reshape(out, shape)

    return out
