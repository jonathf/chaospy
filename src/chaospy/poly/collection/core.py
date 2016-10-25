"""
Collection of tools for manipulation polynomial.
"""

import numpy as np

import chaospy.poly.dimension
import chaospy.bertran

from chaospy.poly.base import Poly


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
        >>> print(cp.basis(4, 4, 2))
        [q0^4, q0^3q1, q0^2q1^2, q0q1^3, q1^4]
        >>> print(cp.basis([1, 1], [2, 2]))
        [q0q1, q0^2q1, q0q1^2, q0^2q1^2]
    """
    if stop is None:
        start, stop = np.array(0), start

    start = np.array(start, dtype=int)
    stop = np.array(stop, dtype=int)
    dim = max(start.size, stop.size, dim)
    indices = np.array(
        chaospy.bertran.bindex(np.min(start), 2*np.max(stop), dim, sort))

    if start.size == 1:
        bellow = np.sum(indices, -1) >= start

    else:
        start = np.ones(dim, dtype=int)*start
        bellow = np.all(indices-start >= 0, -1)

    if stop.size == 1:
        above = np.sum(indices, -1) <= stop.item()

    else:
        stop = np.ones(dim, dtype=int)*stop
        above = np.all(stop-indices >= 0, -1)

    pool = list(indices[above*bellow])

    arg = np.zeros(len(pool), dtype=int)
    arg[0] = 1
    poly = {}
    for idx in pool:
        idx = tuple(idx)
        poly[idx] = arg
        arg = np.roll(arg, 1)

    return Poly(poly, dim)


def cutoff(poly, *args):
    """
    Remove polynomial components with order outside a given interval.

    Args:
        poly (Poly) : Input data.
        low (int, optional) : The lowest order that is allowed to be included.
                Defaults to 0.
        high (int) : The upper threshold for the cutoff range.

    Returns:
        (Poly) : The same as `P`, except that all terms that have a order not
                within the bound `low<=order<high` are removed.

    Examples:
        >>> poly = cp.prange(4, 1) + cp.prange(4, 2)[::-1]
        >>> print(poly)
        [q1^3+1, q1^2+q0, q0^2+q1, q0^3+1]
        >>> print(cp.cutoff(poly, 3))
        [1, q1^2+q0, q0^2+q1, 1]
        >>> print(cp.cutoff(poly, 1, 3))
        [0, q1^2+q0, q0^2+q1, 0]
    """
    if len(args) == 1:
        low, high = 0, args[0]
    else:
        low, high = args[:2]

    core_old = poly.A
    core_new = {}
    for key in poly.keys:
        if low <= np.sum(key) < high:
            core_new[key] = core_old[key]

    return Poly(core_new, poly.dim, poly.shape, poly.dtype)





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
        P = chaospy.poly.dimension.setdim(P, m+1)

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


def all(A, ax=None):
    """ Test if all values in A evaluate to True """
    if isinstance(A, Poly):
        out = np.zeros(A.shape, dtype=bool)
        B = A.A
        for key in A.keys:
            out += all(B[key], ax)
        return out

    return np.all(A, ax)


def any(A, ax=None):
    """ Test if any values in A evaluate to True """
    if isinstance(A, Poly):
        out = np.zeros(A.shape, dtype=bool)
        B = A.A
        for key in A.keys:
            out *= any(B[key])
        return out

    return np.any(A, ax)


def around(A, decimals=0):
    """
    Evenly round to the given number of decimals.

    Args:
        A (Poly, array_like) : Input data.
        decimals (int, optional) : Number of decimal places to round to
                (default: 0).  If decimals is negative, it specifies the number
                of positions to the left of the decimal point.

    Returns:
        (Poly, array_like) : Same type as A.

    Examples:
        >>> P = cp.prange(3)*2**-np.arange(0, 6, 2, float)
        >>> print(P)
        [1.0, 0.25q0, 0.0625q0^2]
        >>> print(cp.around(P))
        [1.0, 0.0, 0.0]
        >>> print(cp.around(P, 2))
        [1.0, 0.25q0, 0.06q0^2]
    """
    if isinstance(A, Poly):
        B = A.A.copy()
        for key in A.keys:
            B[key] = around(B[key], decimals)
        return Poly(B, A.dim, A.shape, A.dtype)

    return np.around(A, decimals)


def diag(A, k=0):
    """ Extract or construct a diagonal polynomial array.  """
    if isinstance(A, Poly):
        core, core_new = A.A, {}
        for key in A.keys:
            core_new[key] = np.diag(core[key], k)

        return Poly(core_new, A.dim, None, A.dtype)

    return np.diag(A, k)


def repeat(A, repeats, axis=None):
    if isinstance(A, Poly):
        core = A.A.copy()
        for key in A.keys:
            core[key] = repeat(core[key], repeats, axis)
        return Poly(core, A.dim, None, A.dtype)

    return np.repeat(A, repeats, axis)


def trace(A, offset=0, ax1=0, ax2=1):
    if isinstance(A, Poly):
        core = A.A.copy()
        for key in A.keys:
            core[key] = trace(core[key], ax1, ax2)
        return Poly(core, A.dim, None, A.dtype)

    return np.trace(A, offset, ax1, ax2)


import chaospy as cp  # pylint: disable=unused-import

if __name__=='__main__':
    import doctest
    doctest.testmod()
