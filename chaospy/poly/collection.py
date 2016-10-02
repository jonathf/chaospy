import numpy as np
from scipy.misc import comb, factorial as fac

import chaospy as cp

from .base import Poly, setdim, decompose, is_decomposed
from chaospy.bertran import terms, multi_index, bindex

__all__ = [
    "basis",
    "cutoff",
    "dot",
    "differential",
    "gradient",
    "hessian",
    "rolldim",
    "swapdim",
    "tril",
    "tricu",
    "variable",
    "order",
    "prange",
]


def basis(start, stop=None, dim=1, sort="G"):
    """
    Create an N-dimensional unit polynomial basis.

    Args:
        start (int, array_like) : the minimum polynomial to include.  If int is
                provided, set as lowest total order.  If array of int, set as
                lower order along each axis.
        stop (int, array_like, optional) : the maximum shape included. If
                omitted: stop <- start; start <- 0 If int is provided, set as
                largest total order.  If array of int, set as largest order
                along each axis.
        dim (int) : dim of the basis.  Ignored if array is provided in either
                start or stop.
        sort (str) : The polynomial ordering where the letters G, I and R can
                be used to set grade, inverse and reverse to the ordering.  For
                `basis(0, 2, 2, order)` we get:
                ------  ------------------
                order   output
                ------  ------------------
                ""      [1 y y^2 x xy x^2]
                "G"     [1 y x y^2 xy x^2]
                "I"     [x^2 xy x y^2 y 1]
                "R"     [1 x x^2 y xy y^2]
                "GIR"   [y^2 xy x^2 y x 1]
                ------  ------------------

    Returns:
        (Poly) : Polynomial array.

    Examples:
        >>> print(cp.basis(4,4,2))
        [q0^4, q0^3q1, q0^2q1^2, q0q1^3, q1^4]
        >>> print(cp.basis([1,1],[2,2]))
        [q0q1, q0^2q1, q0q1^2, q0^2q1^2]
    """
    if stop==None:
        start, stop = 0, start

    start = np.array(start, dtype=int)
    stop = np.array(stop, dtype=int)
    dim = max(start.size, stop.size, dim)
    indices = np.array(bindex(np.min(start), 2*np.max(stop),
        dim, sort))

    if start.size==1:
        bellow = np.sum(indices, -1)>=start.item()
    else:
        start = np.ones(dim, dtype=int)*start
        bellow = np.all(indices-start>=0, -1)

    if stop.size==1:
        above = np.sum(indices, -1)<=stop.item()
    else:
        stop = np.ones(dim, dtype=int)*stop
        above = np.all(stop-indices>=0, -1)

    pool = list(indices[above*bellow])

    x = np.zeros(len(pool), dtype=int)
    x[0] = 1
    A = {}
    for I in pool:
        I = tuple(I)
        A[I] = x
        x = np.roll(x,1)

    return Poly(A, dim)


def lagrange(X):

    X = np.array(X)
    if len(X.shape)<2:
        X = X.reshape(1, *X.shape)
    if len(X.shape)<2:
        X = X.reshape(1, *X.shape)

    dim, K = X.shape

    coefs = np.zeros((dim, K, K))
    for d in range(dim):
        s,t = np.mgrid[:K,:K]
        coefs[d] = X[d,s]-X[d,t]
        coefs[d] += np.eye(K)
    coefs = np.prod(coefs, -1)

    print(coefs)

#  lagrange([(1,2,3),(2,3,4)])
#  fail


def cutoff(P, *args):
    """
    Remove polynomial components with order outside a given interval.

    Args:
        P (Poly) : Input data.
        low (int, optional) : The lowest order that is allowed to be included.
                Defaults to 0.
        high (int) : The upper threshold for the cutoff range.

    Returns:
        (Poly) : The same as `P`, except that all terms that have a order not
                within the bound `low<=order<high` are removed.

    Examples:
        >>> P = prange(4, 1) + prange(4, 2)[::-1]
        >>> print(P)
        [q1^3+1, q1^2+q0, q0^2+q1, q0^3+1]
        >>> print(cutoff(P, 3))
        [1, q1^2+q0, q0^2+q1, 1]
        >>> print(cutoff(P, 1, 3))
        [0, q1^2+q0, q0^2+q1, 0]
    """
    if len(args)==1:
        low, high = 0, args[0]
    else:
        low, high = args[:2]

    A = P.A
    B = {}
    for key in P.keys:
        if low <= np.sum(key) < high:
            B[key] = A[key]

    return Poly(B, P.dim, P.shape, P.dtype)


def dot(P, Q):

    P = Poly(P)
    Q = Poly(Q)
    if np.prod(P.shape)<=1 or np.prod(Q.shape)<=1:
        return P*Q
    return sum(P*Q, -1)


def differential(P, Q):
    """
    Polynomial differential operator.

    Args:
        P (Poly) : Polynomial to be differentiated.
        Q (Poly) : Polynomial to differentiate by. Must be decomposed. If
                polynomial array, the output is the Jacobian matrix.
    """
    P, Q = Poly(P), Poly(Q)

    if not is_decomposed(Q):
        differential(decompose(Q)).sum(0)

    if Q.shape:
        return Poly([differential(P, q) for q in Q])

    if Q.dim>P.dim:
        P = setdim(P, Q.dim)
    else:
        Q = setdim(Q, P.dim)

    qkey = Q.keys[0]

    A = {}
    for key in P.keys:

        newkey = np.array(key) - np.array(qkey)

        if np.any(newkey<0):
            continue

        A[tuple(newkey)] = P.A[key]*np.prod([fac(key[i], \
            exact=True)/fac(newkey[i], exact=True) \
            for i in range(P.dim)])

    return Poly(A, P.dim, None)


def gradient(P):

    return differential(P, basis(1, 1, P.dim))

def hessian(P):

    return gradient(gradient(P))


def prange(N=1, dim=1):
    """
    Constructor to create a range of polynomials where the exponent vary.

    Args:
        N (int) : Number of polynomials in the array.
        dim (int) : The dimension the polynomial should span.

    Returns:
        (Poly) : A polynomial array of length N containing simple polynomials
                with increasing exponent.

    Examples:
        >>> print(prange(4))
        [1, q0, q0^2, q0^3]

        >>> print(prange(4, dim=3))
        [1, q2, q2^2, q2^3]
    """
    A = {}
    r = np.arange(N, dtype=int)
    key = np.zeros(dim, dtype=int)
    for i in range(N):
        key[-1] = i
        A[tuple(key)] = 1*(r==i)

    return Poly(A, dim, (N,), int)


def rolldim(P, n=1):
    """
    Roll the axes.

    Args:
        P (Poly) : Input polynomial.
        n (int) : The axis that after rolling becomes the 0th axis.

    Returns:
        (Poly) : Polynomial with new axis configuration.

    Examples:
        >>> x,y,z = variable(3)
        >>> P = x*x*x + y*y + z
        >>> print(P)
        q0^3+q1^2+q2
        >>> print(rolldim(P))
        q0^2+q2^3+q1
    """
    dim = P.dim
    shape = P.shape
    dtype = P.dtype
    A = dict(((key[n:]+key[:n],P.A[key]) for key in P.keys))
    return Poly(A, dim, shape, dtype)


def swapdim(P, dim1=1, dim2=0):
    """
    Swap the dim between two variables.

    Args:
        P (Poly) : Input polynomial.
        dim1 (int) : First dim
        dim2 (int) : Second dim.

    Returns:
        (Poly) : Polynomial with swapped dimensions.

    Examples
    --------
        >>> x,y = variable(2)
        >>> P = x**4-y
        >>> print(P)
        q0^4-q1
        >>> print(swapdim(P))
        q1^4-q0
    """
    if not isinstance(P, Poly):
        return np.swapaxes(P, dim1, dim2)

    dim = P.dim
    shape = P.shape
    dtype = P.dtype

    if dim1==dim2:
        return P

    m = max(dim1, dim2)
    if P.dim <= m:
        P = setdim(P, m+1)

    A = {}

    for key in P.keys:

        val = P.A[key]
        key = list(key)
        key[dim1], key[dim2] = key[dim2], key[dim1]
        A[tuple(key)] = val

    return Poly(A, dim, shape, dtype)


def tril(P, k=0):
    """Lower triangle of coefficients."""
    A = P.A.copy()
    for key in P.keys:
        A[key] = np.tril(P.A[key])
    return Poly(A, dim=P.dim, shape=P.shape)


def tricu(P, k=0):
    """Cross-diagonal upper triangle."""
    tri = np.sum(np.mgrid[[slice(0,_,1) for _ in P.shape]], 0)
    tri = tri<len(tri) + k

    if isinstance(P, Poly):
        A = P.A.copy()
        B = {}
        for key in P.keys:
            B[key] = A[key]*tri
        return Poly(B, shape=P.shape, dim=P.dim, dtype=P.dtype)

    out = P*tri
    return out


def variable(dims=1):
    """
    Simple constructor to create single variables to create polynomials.

    Args:
        dims (int) : Number of dimensions in the array.

    Returns:
        (Poly) : Polynomial array with unit components in each dimension.

    Examples:
        >>> print(variable())
        q0
        >>> print(variable(3))
        [q0, q1, q2]
    """
    if dims==1:
        return Poly({(1,):np.array(1)}, dim=1, shape=())

    r = np.arange(dims, dtype=int)
    A = {}
    for i in range(dims):
        A[tuple(1*(r==i))] = 1*(r==i)

    return Poly(A, dim=dims, shape=(dims,))

def order(P):

    out = np.zeros(P.shape, dtype=int)
    for key in P.keys:
        o = sum(key)
        out = np.max([out, o*(P.A[key])], 0)
    return out


# def roots(P, ax=0, args=None):
#     """
# Find the roots of a polynomial or construct a
# polynomials from roots.
# 
# Parameters
# ----------
# P : Poly, array_like
#     The polynomial or roots in question. If Poly is provided,
#     roots will be returned and vice versa.
# ax : int
#     Axis which the roots are found. If polynomial has more
#     then one dimensions, roots are taken along a given axes
#     ax. The remaining axes are evaluated.
# args : array_like
#     Arguments for the axes to be evaluated to create a one
#     dimensional polynomial. Value in position ax-1 is ignored.
#     evaluate all values as 1 if omitted.
#     
# Returns
# -------
# Q : ndarray, Poly
#     List of roots or Poly dependent on P.
# 
# Examples
# --------
# Find roots of polynomial
# 
# >>> x = variable()
# >>> print(roots(x*x-1))
# [-1.  1.]
# 
# Find polynomials from roots
# >>> print(roots([-1,1]))
# x^2-1
# 
# Roots along an axis
# >>> x,y = variable(2)
# >>> P = (x*x-1)*(y-2)
# >>> print(roots(P))
# [ 1. -1.]
# >>> print(roots(P, ax=1, args=[2,0]))
# [ 2.]
#     """
# 
#     if not isinstance(P, Poly):
#         A = {}
#         coefs = np.poly(P)[::-1]
#         for i in range(len(coefs)):
#             if coefs[i]:
#                 A[(i,)] = np.array(coefs[i])
#         return Poly(A)
# 
#     if P.dim>1:
#         if args==None:
#             args = [1]*P.dim
#         else:
#             args = list(args)
#         args[ax] = np.nan
#         P = P(*args)
#         P = swapdim(P, 0, ax)
#         P = setdim(P, 1)
# 
#     coef = []
#     P.keys.sort(key=lambda x: sum(x)**P.dim +\
#         sum(x*P.dim**np.arange(P.dim)),reverse=1)
#     length = P.keys[0][0]+1
#     for key in range(length):
#         if P.A.has_key((key,)):
#             coef.append(P.A[(key,)])
#         else:
#             coef.append(0)
#     coef = np.array(coef).flatten()
#     return np.roots(coef[::-1])


if __name__=='__main__':
    import __init__ as cp
    import doctest
    doctest.testmod()
