import numpy as np

import chaospy as cp

def Var(poly, dist=None, **kws):
    """
    Element by element 2nd order statistics.

    Args:
        poly (Poly, Dist) : Input to take variance on.
        dist (Dist) : Defines the space the variance is taken on. It is ignored
                if `poly` is a distribution.
        **kws (optional) : Extra keywords passed to dist.mom.

    Returns:
        (ndarray) : Element for element variance along `poly`, where
                `variation.shape==poly.shape`.

    Examples:
        >>> x = cp.variable()
        >>> Z = cp.Uniform()
        >>> print(np.around(cp.Var(Z), 8))
        0.08333333
        >>> print(np.around(cp.Var(x**3, Z), 8))
        0.08035714
    """
    if isinstance(poly, cp.dist.Dist):
        x = cp.poly.variable(len(poly))
        poly, dist = x, poly
    else:
        poly = cp.poly.Poly(poly)

    dim = len(dist)
    if poly.dim<dim:
        cp.poly.setdim(poly, dim)

    shape = poly.shape
    poly = cp.poly.flatten(poly)

    keys = poly.keys
    N = len(keys)
    A = poly.A

    keys1 = np.array(keys).T
    if dim==1:
        keys1 = keys1[0]
        keys2 = sum(np.meshgrid(keys, keys))
    else:
        keys2 = np.empty((dim, N, N))
        for i in range(N):
            for j in range(N):
                keys2[:, i, j] = keys1[:, i]+keys1[:, j]

    m1 = np.outer(*[dist.mom(keys1, **kws)]*2)
    m2 = dist.mom(keys2, **kws)
    mom = m2-m1

    out = np.zeros(poly.shape)
    for i in range(N):
        a = A[keys[i]]
        out += a*a*mom[i, i]
        for j in range(i+1, N):
            b = A[keys[j]]
            out += 2*a*b*mom[i, j]

    out = out.reshape(shape)
    return out


def Std(poly, dist=None, **kws):
    """
    Element by element 2nd order statistics.

    Args:
        poly (Poly, Dist) : Input to take variance on.
        dist (Dist) : Defines the space the variance is taken on.
                It is ignored if `poly` is a distribution.
        **kws (optional) : Extra keywords passed to dist.mom.

    Returns:
        (ndarray) : Element for element variance along `poly`, where
                `variation.shape==poly.shape`.

    Examples:
        >>> x = cp.variable()
        >>> Z = cp.Uniform()
        >>> print(np.around(cp.Var(Z), 8))
        0.08333333
        >>> print(np.around(cp.Var(x**3, Z), 8))
        0.08035714
    """

    if isinstance(poly, cp.dist.Dist):
        x = cp.poly.variable(len(poly))
        poly, dist = x, poly
    else:
        poly = cp.poly.Poly(poly)

    dim = len(dist)
    if poly.dim<dim:
        cp.poly.setdim(poly, dim)

    shape = poly.shape
    poly = cp.poly.flatten(poly)

    keys = poly.keys
    N = len(keys)
    A = poly.A

    keys1 = np.array(keys).T
    if dim==1:
        keys1 = keys1[0]
        keys2 = sum(np.meshgrid(keys, keys))
    else:
        keys2 = np.empty((dim, N, N))
        for i in range(N):
            for j in range(N):
                keys2[:, i, j] = keys1[:, i]+keys1[:, j]

    m1 = np.outer(*[dist.mom(keys1, **kws)]*2)
    m2 = dist.mom(keys2, **kws)
    mom = m2-m1

    out = np.zeros(poly.shape)
    for i in range(N):
        a = A[keys[i]]
        out += a*a*mom[i, i]
        for j in range(i+1, N):
            b = A[keys[j]]
            out += 2*a*b*mom[i, j]

    out = out.reshape(shape)
    return np.sqrt(out)
