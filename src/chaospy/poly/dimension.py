import numpy as np

import chaospy
from chaospy.poly.base import Poly, sort_key


def dimsplit(P):
    """
    Segmentize a polynomial (on decomposed form) into it's dimensions.

    In array missing values are padded with 1 to make dimsplit compatible with
    `poly.prod(Q, 0)`.


    Args:
        P (Poly) : Input polynomial.

    Returns:
        (Poly) : Segmentet polynomial array where
                `Q.shape==P.shape+(P.dim+1,)`. The surplus element in `P.dim+1`
                is used for coeficients.

    Examples:
        >>> x,y = cp.variable(2)
        >>> P = cp.Poly([2, x*y, 2*x])
        >>> Q = cp.dimsplit(P)
        >>> print(Q)
        [[2, 1, 2], [1, q0, q0], [1, q1, 1]]
        >>> print(cp.prod(Q, 0))
        [2, q0q1, 2q0]
    """
    P = P.copy()

    if not chaospy.poly.caller.is_decomposed(P):
        raise TypeError("Polynomial not on component form.")
    A = []

    dim = P.dim
    coef = P(*(1,)*dim)
    M = coef!=0
    zero = (0,)*dim
    ones = [1]*dim
    A = [{zero: coef}]

    if zero in P.A:

        del P.A[zero]
        P.keys.remove(zero)

    for key in P.keys:
        P.A[key] = (P.A[key]!=0)

    for i in range(dim):

        A.append({})
        ones[i] = np.nan
        Q = P(*ones)
        ones[i] = 1
        if isinstance(Q, np.ndarray):
            continue
        Q = Q.A

        if zero in Q:
            del Q[zero]

        for key in Q:

            val = Q[key]
            A[-1][key] = val

    A = [Poly(a, dim, None, P.dtype) for a in A]
    P = Poly(A, dim, None, P.dtype)
    P = P + 1*(P(*(1,)*dim)==0)*M

    return P


def setdim(P, dim=None):
    """
    Adjust the dimensions of a polynomial.

    Output the results into Poly object

    Args:
        P (Poly) : Input polynomial
        dim (int) : The dimensions of the output polynomial. If omitted,
                increase polynomial with one dimension. If the new dim is
                smaller then P's dimensions, variables with cut components are
                all cut.

    Examples:
        >>> x,y = cp.variable(2)
        >>> P = x*x-x*y
        >>> print(cp.setdim(P, 1))
        q0^2
    """
    P = P.copy()

    ldim = P.dim
    if not dim:
        dim = ldim+1

    if dim==ldim:
        return P

    P.dim = dim
    if dim>ldim:

        key = np.zeros(dim, dtype=int)
        for lkey in P.keys:
            key[:ldim] = lkey
            P.A[tuple(key)] = P.A.pop(lkey)

    else:

        key = np.zeros(dim, dtype=int)
        for lkey in P.keys:
            if not sum(lkey[ldim-1:]) or not sum(lkey):
                P.A[lkey[:dim]] = P.A.pop(lkey)
            else:
                del P.A[lkey]

    P.keys = sorted(P.A.keys(), key=sort_key)
    return P


import chaospy as cp
