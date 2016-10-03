import numpy as np
import chaospy as cp

from . import base as p
from . import fraction as f
SUM = sum

__all__ = [
    "all",
    "any",
    "around",
    "asfloat",
    "asfrac",
    "asint",
    "cumprod",
    "cumsum",
    "diag",
    "dtyping",
    "flatten",
    "inner",
    "mean",
    "outer",
    "prod",
    "repeat",
    "reshape",
    "roll",
    "rollaxis",
    "std",
    "sum",
    "swapaxes",
    "toarray",
    "trace",
    "transpose",
    "var",
]

def all(A, ax=None):
    """ Test if all values in A evaluate to True """

    if isinstance(A, (np.ndarray, float, int)):
        return np.all(A, ax)

    elif isinstance(A, f.frac):
        return np.all(A.a, ax)

    elif isinstance(A, p.Poly):

        out = np.zeros(A.shape, dtype=bool)
        B = A.A
        for key in A.keys:
            out += all(B[key], ax)
        return out

    raise NotImplementedError

def any(A, ax=None):
    """ Test if any values in A evaluate to True """

    if isinstance(A, (np.ndarray, float, int)):
        return np.any(A, ax)

    elif isinstance(A, f.frac):
        return np.any(A.a, ax)

    elif isinstance(A, p.Poly):
        out = np.zeros(A.shape, dtype=bool)
        B = A.A
        for key in A.keys:
            out *= any(B[key])
        return out

    raise NotImplementedError


def around(A, decimals=0):
    """
    Evenly round to the given number of decimals.

    Args:
        A (p.Poly, f.frac, array_like) : Input data.
        decimals (int, optional) : Number of decimal places to round to
                (default: 0).  If decimals is negative, it specifies the number
                of positions to the left of the decimal point.

    Returns:
        (p.Poly, f.frac, array_like) : Same type as A.

    Examples:
        >>> P = cp.prange(3)*2**-np.arange(0, 6, 2, float)
        >>> print(P)
        [1.0, 0.25q0, 0.0625q0^2]
        >>> print(cp.around(P))
        [1.0, 0.0, 0.0]
        >>> print(cp.around(P, 2))
        [1.0, 0.25q0, 0.06q0^2]
    """
    if isinstance(A, (np.ndarray, float, int)):
        return np.around(A, decimals)

    elif isinstance(A, f.frac):

        if decimals>=0:
            a = (A.a*10**decimals)//A.b
            return f.frac(a, 10**decimals)
        a = A.a//(A.b*10**-decimals)
        return f.frac(a*10**-decimals)

    elif isinstance(A, p.Poly):

        B = A.A.copy()
        for key in A.keys:
            B[key] = around(B[key], decimals)
        return p.Poly(B, A.dim, A.shape, A.dtype)

    raise NotImplementedError


def asfloat(A):

    if isinstance(A, (np.ndarray, float, int)):
        return np.asfarray(A)

    elif isinstance(A, f.frac):
        return f.asfloat(A)

    elif isinstance(A, p.Poly):
        return p.asfloat(A)

    raise NotImplementedError


def reshape(a, shape):
    """
    Reshape the shape of a shapeable quantity.

    Args:
        a (Poly, frac, array_like) : Shapeable input quantity
        shape (tuple) : The polynomials new shape. Must be compatible with the
                number of elements in `A`.

    Returns:
        (Poly, frac, array_like) : Same type as `A`.

    Examples:
        >>> P = cp.prange(6)
        >>> print(P)
        [1, q0, q0^2, q0^3, q0^4, q0^5]
        >>> print(cp.reshape(P, (2,3)))
        [[1, q0, q0^2], [q0^3, q0^4, q0^5]]
    """
    if isinstance(a, (np.ndarray, float, int)):
        return np.reshape(a, shape)

    elif isinstance(a, f.frac):
        return f.reshape(a, shape)

    elif isinstance(a, p.Poly):
        return p.reshape(a, shape)
    raise NotImplementedError


def flatten(A):
    """
    Flatten a shapeable quantity.

    Args:
        A (Poly, frac, array_like) : Shapeable input quantity.

    Returns:
        (Poly, frac, array_like) : Same type as `A` with `len(Q.shape)==1`.

    Examples:
        >>> P = cp.reshape(cp.prange(4), (2,2))
        >>> print(P)
        [[1, q0], [q0^2, q0^3]]
        >>> print(cp.flatten(P))
        [1, q0, q0^2, q0^3]
    """
    if isinstance(A, (np.ndarray, float, int)):
        return np.flatten(A)

    elif isinstance(A, f.frac):
        return f.flatten(A)

    elif isinstance(A, p.Poly):
        return p.flatten(A)
    raise NotImplementedError


def sum(A, axis=None):
    """
    Sum the components of a shapeable quantity along a given axis.

    Args:
        A (Poly, frac, array_like) : Input data.
        axis (int, optional) : Axis over which the sum is taken. By default
                `axis` is None, and all elements are summed.

    Returns:
        (Poly, frac, array_like) : Polynomial array with same shape as `P`,
                with the specified axis removed. If `P` is an 0-d array, or
                `axis` is None, a (non-iterable) component is returned.

    Examples:
        >>> P = cp.prange(3)
        >>> print(P)
        [1, q0, q0^2]
        >>> print(cp.sum(P))
        q0^2+q0+1
    """
    if isinstance(A, (np.ndarray, float, int)):
        return np.sum(A, axis)

    elif isinstance(A, f.frac):
        return f.sum(A, axis)

    elif isinstance(A, p.Poly):
        return p.sum(A, axis)

    raise NotImplementedError


def prod(A, axis=None):
    """
    Perform the product of a shapeable quantity over a given axis.

    Args:
        P (Poly, frac, array_like) : Input data.
        axis (int, optional) : Axis over which the product is taken.  By
                default, the product of all elements is calculated.

    Returns:
        (Poly) : An array shaped as `A` but with the specified axis removed.

    Examples:
        >>> P = cp.reshape(cp.prange(8), (2,2,2))
        >>> print(P)
        [[[1, q0], [q0^2, q0^3]], [[q0^4, q0^5], [q0^6, q0^7]]]
        >>> print(cp.prod(P))
        q0^28
        >>> print(cp.prod(P, 0))
        [[q0^4, q0^6], [q0^8, q0^10]]
        >>> print(cp.prod(P, -1))
        [[q0, q0^5], [q0^9, q0^13]]
    """
    if isinstance(A, (np.ndarray, float, int)):
        return np.prod(A, axis)

    elif isinstance(A, f.frac):
        return f.prod(A, axis)

    elif isinstance(A, p.Poly):
        return p.prod(A, axis)

    raise NotImplementedError

def asfrac(A, limit=None):

    if isinstance(A, (np.ndarray, float, int)):
        return f.frac(A, 1, limit)

    elif isinstance(A, f.frac):
        return f.frac(A.a, A.b, limit)

    elif isinstance(A, p.Poly):
        return p.asfrac(A, limit)

    raise NotImplementedError

def asint(A):

    if isinstance(A, (np.ndarray, float, int)):
        return np.array(A, dtype=int)

    elif isinstance(A, f.frac):
        return f.asint(A)

    elif isinstance(A, p.Poly):
        return p.asint(A)

    raise NotImplementedError


def toarray(A):
    """
    Convert to a numpy.array object.

    Args:
        A (p.Poly, f.frac, array_like) : Input data.

    Returns:
        Q (ndarray) : A numpy.ndarray with `Q.shape==A.shape`.

    Examples:
        >>> P = cp.prange(4)
        >>> Q = cp.toarray(P)
        >>> print(isinstance(Q, np.ndarray))
        True
        >>> print(Q[1] == P[1])
        True
    """
    if isinstance(A, (np.ndarray, float, int)):
        return np.array(A)

    elif isinstance(A, f.frac):
        return f.toarray(A)

    elif isinstance(A, p.Poly):
        return p.toarray(A)

    raise NotImplementedError


def mean(A, ax=None):

    if isinstance(A, (int, float, np.ndarray)):
        return np.mean(A, ax)

    elif isinstance(A, f.frac):
        return f.mean(A)

    elif isinstance(A, p.Poly):
        return p.mean(A)

    raise NotImplementedError

def var(A, ax=None):

    if isinstance(A, (int, float, np.ndarray)):
        return np.var(A, ax)

    elif isinstance(A, f.frac):
        return f.var(A)

    elif isinstance(A, p.Poly):
        return p.var(A)

    raise NotImplementedError

def transpose(A):
    """
    Transpose a shapeable quantety.

    Args:
        A (Poly, frac, array_like) : Quantety of interest.

    Returns:
        Q (Poly, frac, array_like) : Same type as `A`.

    Examples:
        >>> P = cp.reshape(cp.prange(4), (2,2))
        >>> print(P)
        [[1, q0], [q0^2, q0^3]]

        >>> print(cp.transpose(P))
        [[1, q0^2], [q0, q0^3]]
    """

    if isinstance(A, (int, float, np.ndarray)):
        return np.transpose(A)

    elif isinstance(A, f.frac):
        return f.transpose(A)

    elif isinstance(A, p.Poly):
        return p.transpose(A)

    raise NotImplementedError


def rollaxis(A, ax, start=0):

    if isinstance(A, (int, float, np.ndarray)):
        return np.rollaxis(A, ax, start)

    elif isinstance(A, f.frac):
        return f.rollaxis(A, ax, start)

    elif isinstance(A, p.Poly):
        return p.rollaxis(A, ax, start)

    raise NotImplementedError


def roll(A, shift, axis=None):

    if isinstance(A, (int, float, np.ndarray)):
        return np.roll(A, shift, axis)

    elif isinstance(A, f.frac):
        return f.roll(A, shift, axis)

    elif isinstance(A, p.Poly):
        return p.roll(A, shift, axis)

    raise NotImplementedError


def cumsum(A, axis=None):

    if isinstance(A, (int, float, np.ndarray)):
        return np.cumsum(A, axis)

    elif isinstance(A, f.frac):
        return f.cumsum(A, axis)

    elif isinstance(A, p.Poly):
        return p.cumsum(A, axis)

    raise NotImplementedError


def cumprod(A, axis=None):

    if isinstance(A, (int, float, np.ndarray)):
        return np.cumprod(A, axis)

    elif isinstance(A, f.frac):
        return f.cumprod(A, axis)

    elif isinstance(A, p.Poly):
        return p.cumprod(A, axis)

    raise NotImplementedError


def diag(A, k=0):
    """ Extract or construct a diagonal polynomial array.  """

    if isinstance(A, (int, float, np.ndarray)):
        return np.diag(A, k)

    elif isinstance(A, f.frac):
        return f.diag(A, k)

    elif isinstance(A, p.Poly):
        return p.diag(A, k)

    raise NotImplementedError

def repeat(A, repeats, axis=None):

    if isinstance(A, (int, float, np.ndarray)):
        return np.repeat(A, repeats, axis)

    elif isinstance(A, f.frac):
        return f.repeat(A, repeats, axis)

    elif isinstance(A, p.Poly):
        return p.repeat(A, repeats, axis)

    raise NotImplementedError

def std(A, axis):

    if isinstance(A, (int, float, np.ndarray)):
        return np.std(A, axis)

    elif isinstance(A, f.frac):
        return f.std(A, axis)

    elif isinstance(A, p.Poly):
        return p.std(A, axis)

    raise NotImplementedError


def swapaxes(A, ax1, ax2):
    if isinstance(A, (int, float, np.ndarray)):
        return np.swapaxes(A, ax1, ax2)

    elif isinstance(A, f.frac):
        return f.swapaxes(A, ax1, ax2)

    elif isinstance(A, p.Poly):
        return p.swapaxes(A, ax1, ax2)

    raise NotImplementedError


def trace(A, offset=0, ax1=0, ax2=1):

    if isinstance(A, (int, float, np.ndarray)):
        return np.trace(A, offset, ax1, ax2)

    elif isinstance(A, f.frac):
        return f.trace(A, offset, ax1, ax2)

    elif isinstance(A, p.Poly):
        return p.trace(A, offset, ax1, ax2)

    raise NotImplementedError


def dtyping(*args):
    return p.dtyping(*args)


def inner(*args):
    """
    Inner product of a polynomial set.

    If no p.Poly object provided, numpy.inner used.

    Args:
        arg0, arg1, [...] (Poly) : The polynomials to perform inner product on.

    Returns:
        (Poly) : Resulting polynomial.

    Examples:
        >>> x,y = cp.variable(2)
        >>> P = cp.Poly([x-1, y])
        >>> Q = cp.Poly([x+1, x*y])
        >>> print(cp.inner(P, Q))
        q0^2+q0q1^2-1
        >>> x = np.arange(4)
        >>> print(cp.inner(x, x))
        14
    """

    haspoly = SUM([isinstance(arg, p.Poly) for arg in args])
    hasfrac = SUM([isinstance(arg, f.frac) for arg in args])

    # Numpy
    if not haspoly and not hasfrac:
        return np.sum(np.prod(args, 0), 0)

    # Frac
    if not haspoly:
        args = map(f.frac, args)
        return f.inner(*args)

    # Poly
    args = map(p.Poly, args)
    return p.inner(*args)

def outer(*args):
    """
    Polynomial outerproduct.

    Args:
        P1 (Poly, ndarray, int, float) : First term in outer product
        P2 (Poly, array_like) : Second term in outer product

    Returns:
        (Poly) : Poly set with same dimensions as itter.

    Examples:
        >>> x = cp.variable()
        >>> P = cp.prange(3)
        >>> print(P)
        [1, q0, q0^2]
        >>> print(cp.outer(x, P))
        [q0, q0^2, q0^3]
        >>> print(cp.outer(P, P))
        [[1, q0, q0^2], [q0, q0^2, q0^3], [q0^2, q0^3, q0^4]]
    """
    dtype = dtyping(*map(type, args))

    if dtype==p.Poly:
        return p.outer(*args)

    if dtype==f.frac:
        return f.outer(*map(p.frac, args))

    if dtype in (list, tuple, np.ndarray):

        if len(args)==1:
            return outer(*args[0])

        dtype = dtyping(*[_.dtype \
            for _ in map(np.array, args)])

        if dtype==float:
            args = map(asfloat, args)
        elif dtype==object:
            args = map(asfrac, args)
        elif dtype in (int):
            args = map(asint, args)

        if len(args)>2:
            P1 = args[0]
            P2 = outer(*args[1:])
        elif len(args)==2:
            P1,P2 = args
        else:
            return args[0]

        shape1 = P1.shape
        shape2 = P2.shape
        out = np.outer(P1.flatten(), P2.flatten())
        return out.reshape(shape1+shape2)

    raise NotImplementedError


if __name__=='__main__':
    import doctest
    doctest.testmod()
