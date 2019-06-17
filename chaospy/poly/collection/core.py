"""
Collection of tools for manipulation polynomial.
"""

import numpy
from scipy.special import comb, factorial as fac

import chaospy.poly.dimension
import chaospy.bertran
import chaospy.quad

from chaospy.poly.base import Poly


def basis(start, stop=None, dim=1, sort="G", cross_truncation=1.):
    """
    Create an N-dimensional unit polynomial basis.

    Args:
        start (int, numpy.ndarray):
            the minimum polynomial to include. If int is provided, set as
            lowest total order.  If array of int, set as lower order along each
            axis.
        stop (int, numpy.ndarray):
            the maximum shape included. If omitted:
            ``stop <- start; start <- 0`` If int is provided, set as largest
            total order. If array of int, set as largest order along each axis.
        dim (int):
            dim of the basis. Ignored if array is provided in either start or
            stop.
        sort (str):
            The polynomial ordering where the letters ``G``, ``I`` and ``R``
            can be used to set grade, inverse and reverse to the ordering.  For
            ``basis(start=0, stop=2, dim=2, order=order)`` we get:
            ======  ==================
            order   output
            ======  ==================
            ""      [1 y y^2 x xy x^2]
            "G"     [1 y x y^2 xy x^2]
            "I"     [x^2 xy x y^2 y 1]
            "R"     [1 x x^2 y xy y^2]
            "GIR"   [y^2 xy x^2 y x 1]
            ======  ==================
        cross_truncation (float):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion.

    Returns:
        (Poly) : Polynomial array.

    Examples:
        >>> print(chaospy.basis(4, 4, 2, sort="GR"))
        [q0^4, q0^3q1, q0^2q1^2, q0q1^3, q1^4]
        >>> print(chaospy.basis([1, 1], [2, 2], sort="GR"))
        [q0q1, q0^2q1, q0q1^2, q0^2q1^2]
    """
    if stop is None:
        start, stop = numpy.array(0), start

    start = numpy.array(start, dtype=int)
    stop = numpy.array(stop, dtype=int)
    dim = max(start.size, stop.size, dim)
    indices = numpy.array(chaospy.bertran.bindex(
        numpy.min(start), 2*numpy.max(stop), dim, sort, cross_truncation))

    if start.size == 1:
        bellow = numpy.sum(indices, -1) >= start

    else:
        start = numpy.ones(dim, dtype=int)*start
        bellow = numpy.all(indices-start >= 0, -1)


    if stop.size == 1:
        above = numpy.sum(indices, -1) <= stop.item()
    else:
        stop = numpy.ones(dim, dtype=int)*stop
        above = numpy.all(stop-indices >= 0, -1)

    pool = list(indices[above*bellow])

    arg = numpy.zeros(len(pool), dtype=int)
    arg[0] = 1
    poly = {}
    for idx in pool:
        idx = tuple(idx)
        poly[idx] = arg
        arg = numpy.roll(arg, 1)
    x = numpy.zeros(len(pool), dtype=int)
    x[0] = 1
    A = {}
    for I in pool:
        I = tuple(I)
        A[I] = x
        x = numpy.roll(x,1)

    return Poly(A, dim)


def lagrange(X):

    X = numpy.array(X)
    if len(X.shape) < 2:
        X = X.reshape(1, *X.shape)
    if len(X.shape) < 2:
        X = X.reshape(1, *X.shape)

    dim, K = X.shape

    return Poly(poly, dim)


def cutoff(poly, *args):
    """
    Remove polynomial components with order outside a given interval.

    Args:
        poly (Poly):
            Input data.
        low (int):
            The lowest order that is allowed to be included. Defaults to 0.
        high (int):
            The upper threshold for the cutoff range.

    Returns:
        (Poly):
            The same as `P`, except that all terms that have a order not within
            the bound `low <= order < high` are removed.

    Examples:
        >>> poly = chaospy.prange(4, 1) + chaospy.prange(4, 2)[::-1]
        >>> print(poly) # doctest: +SKIP
        [q1^3+1, q0+q1^2, q0^2+q1, q0^3+1]
        >>> print(chaospy.cutoff(poly, 3)) # doctest: +SKIP
        [1, q0+q1^2, q0^2+q1, 1]
        >>> print(chaospy.cutoff(poly, 1, 3)) # doctest: +SKIP
        [0, q0+q1^2, q0^2+q1, 0]
    """
    if len(args) == 1:
        low, high = 0, args[0]
    else:
        low, high = args[:2]

    core_old = poly.A
    core_new = {}
    for key in poly.keys:
        if low <= numpy.sum(key) < high:
            core_new[key] = core_old[key]
    return Poly(core_new, poly.dim, poly.shape, poly.dtype)


def dot(P, Q):

    P = Poly(P)
    Q = Poly(Q)
    if numpy.prod(P.shape)<=1 or numpy.prod(Q.shape)<=1:
        return P*Q
    return sum(P*Q, -1)


def differential(P, Q):
    """
    Polynomial differential operator.

    Args:
        P (Poly):
            Polynomial to be differentiated.
        Q (Poly):
            Polynomial to differentiate by. Must be decomposed. If polynomial
            array, the output is the Jacobian matrix.
    """
    P, Q = Poly(P), Poly(Q)

    if not chaospy.poly.is_decomposed(Q):
        differential(chaospy.poly.decompose(Q)).sum(0)

    if Q.shape:
        return Poly([differential(P, q) for q in Q])

    if Q.dim>P.dim:
        P = chaospy.poly.setdim(P, Q.dim)
    else:
        Q = chaospy.poly.setdim(Q, P.dim)

    qkey = Q.keys[0]

    A = {}
    for key in P.keys:

        newkey = numpy.array(key) - numpy.array(qkey)

        if numpy.any(newkey<0):
            continue

        A[tuple(newkey)] = P.A[key]*numpy.prod([fac(key[i], \
            exact=True)/fac(newkey[i], exact=True) \
            for i in range(P.dim)])

    return Poly(B, P.dim, P.shape, P.dtype)


def prange(N=1, dim=1):
    """
    Constructor to create a range of polynomials where the exponent vary.

    Args:
        N (int):
            Number of polynomials in the array.
        dim (int):
            The dimension the polynomial should span.

    Returns:
        (Poly):
            A polynomial array of length N containing simple polynomials with
            increasing exponent.

    Examples:
        >>> print(prange(4))
        [1, q0, q0^2, q0^3]
        >>> print(prange(4, dim=3))
        [1, q2, q2^2, q2^3]
    """
    A = {}
    r = numpy.arange(N, dtype=int)
    key = numpy.zeros(dim, dtype=int)
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
        P (Poly):
            Input polynomial.
        dim1 (int):
            First dim
        dim2 (int):
            Second dim.

    Returns:
        (Poly):
            Polynomial with swapped dimensions.

    Examples:
        >>> x,y = variable(2)
        >>> P = x**4-y
        >>> print(P)
        q0^4-q1
        >>> print(swapdim(P))
        q1^4-q0
    """
    if not isinstance(P, Poly):
        return numpy.swapaxes(P, dim1, dim2)

    dim = P.dim
    shape = P.shape
    dtype = P.dtype

    if dim1==dim2:
        return P

    m = max(dim1, dim2)
    if P.dim <= m:
        P = chaospy.poly.dimension.setdim(P, m+1)
        dim = m+1

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
        A[key] = numpy.tril(P.A[key])
    return Poly(A, dim=P.dim, shape=P.shape)


def tricu(P, k=0):
    """Cross-diagonal upper triangle."""
    tri = numpy.sum(numpy.mgrid[[slice(0,_,1) for _ in P.shape]], 0)
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
        dims (int):
            Number of dimensions in the array.

    Returns:
        (Poly):
            Polynomial array with unit components in each dimension.

    Examples:
        >>> print(variable())
        q0
        >>> print(variable(3))
        [q0, q1, q2]
    """
    if dims == 1:
        return Poly({(1,): 1}, dim=1, shape=())
    return Poly({
        tuple(indices): indices for indices in numpy.eye(dims, dtype=int)
    }, dim=dims, shape=(dims,))

def order(P):

    out = numpy.zeros(P.shape, dtype=int)
    for key in P.keys:
        o = sum(key)
        out = numpy.max([out, o*(P.A[key])], 0)
    return out


def all(A, ax=None):
    """Test if all values in A evaluate to True """
    if isinstance(A, Poly):
        out = numpy.zeros(A.shape, dtype=bool)
        B = A.A
        for key in A.keys:
            out += all(B[key], ax)
        return out

    return numpy.all(A, ax)


def any(A, ax=None):
    """Test if any values in A evaluate to True """
    if isinstance(A, Poly):
        out = numpy.zeros(A.shape, dtype=bool)
        B = A.A
        for key in A.keys:
            out *= any(B[key])
        return out

    return numpy.any(A, ax)


def around(A, decimals=0):
    """
    Evenly round to the given number of decimals.

    Args:
        A (Poly, numpy.ndarray):
            Input data.
        decimals (int):
            Number of decimal places to round to (default: 0).  If decimals is
            negative, it specifies the number of positions to the left of the
            decimal point.

    Returns:
        (Poly, numpy.ndarray):
            Same type as A.

    Examples:
        >>> P = chaospy.prange(3)*2**-numpy.arange(0, 6, 2, float)
        >>> print(P)
        [1.0, 0.25q0, 0.0625q0^2]
        >>> print(chaospy.around(P))
        [1.0, 0.0, 0.0]
        >>> print(chaospy.around(P, 2))
        [1.0, 0.25q0, 0.06q0^2]
    """
    if isinstance(A, Poly):
        B = A.A.copy()
        for key in A.keys:
            B[key] = around(B[key], decimals)
        return Poly(B, A.dim, A.shape, A.dtype)

    return numpy.around(A, decimals)


def diag(A, k=0):
    """Extract or construct a diagonal polynomial array."""
    if isinstance(A, Poly):
        core, core_new = A.A, {}
        for key in A.keys:
            core_new[key] = numpy.diag(core[key], k)

        return Poly(core_new, A.dim, None, A.dtype)

    return numpy.diag(A, k)


def repeat(A, repeats, axis=None):
    if isinstance(A, Poly):
        core = A.A.copy()
        for key in A.keys:
            core[key] = repeat(core[key], repeats, axis)
        return Poly(core, A.dim, None, A.dtype)

    return numpy.repeat(A, repeats, axis)


def trace(A, offset=0, ax1=0, ax2=1):
    if isinstance(A, Poly):
        core = A.A.copy()
        for key in A.keys:
            core[key] = trace(core[key], ax1, ax2)
        return Poly(core, A.dim, None, A.dtype)

    return numpy.trace(A, offset, ax1, ax2)


def prune(A, threshold):
    """
    Remove coefficients that is not larger than a given threshold.

    Args:
        A (Poly):
            Input data.
        threshold (float):
            Threshold for which values to cut.

    Returns:
        (Poly):
            Same type as A.

    Examples:
        >>> P = chaospy.sum(chaospy.prange(3)*2**-numpy.arange(0, 6, 2, float))
        >>> print(P)
        0.0625q0^2+0.25q0+1.0
        >>> print(chaospy.prune(P, 0.1))
        0.25q0+1.0
        >>> print(chaospy.prune(P, 0.5))
        1.0
        >>> print(chaospy.prune(P, 1.5))
        0.0
    """
    if isinstance(A, Poly):
        B = A.A.copy()
        for key in A.keys:
            values = B[key].copy()
            values[numpy.abs(values) < threshold] = 0.
            B[key] = values
        return Poly(B, A.dim, A.shape, A.dtype)

    A = A.copy()
    A[numpy.abs(A) < threshold] = 0.
    return A
