"""
Functions mimicing linear algebra operations.
"""

import numpy as np
import chaospy as cp

import chaospy.poly
from chaospy.poly.base import Poly


def inner(*args):
    """
    Inner product of a polynomial set.

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
    haspoly = sum([isinstance(arg, Poly) for arg in args])

    # Numpy
    if not haspoly:
        return np.sum(np.prod(args, 0), 0)

    # Poly
    out = args[0]
    for arg in args[1:]:
        out = out * arg
    return sum(out)


def outer(*args):
    """
    Polynomial outer product.

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
    if len(args) > 2:
        part1 = args[0]
        part2 = outer(*args[1:])

    elif len(args) == 2:
        part1, part2 = args

    else:
        return args[0]

    dtype = chaospy.poly.typing.dtyping(part1, part2)

    if dtype in (list, tuple, np.ndarray):

        part1 = np.array(part1)
        part2 = np.array(part2)
        shape = part1.shape +  part2.shape
        return np.outer(
            chaospy.poly.shaping.flatten(part1),
            chaospy.poly.shaping.flatten(part2),
        )

    if dtype == Poly:

        if isinstance(part1, Poly) and isinstance(part2, Poly):

            if (1,) in (part1.shape, part2.shape):
                return part1*part2

            shape = part1.shape+part2.shape

            out = []
            for _ in chaospy.poly.shaping.flatten(part1):
                out.append(part2*_)

            return chaospy.poly.shaping.reshape(Poly(out), shape)

        if isinstance(part1, (int, float, list, tuple)):
            part2, part1 = np.array(part1), part2

        else:
            part2 = np.array(part2)

        core_old = part1.A
        core_new = {}
        for key in part1.keys:
            core_new[key] = outer(core_old[key], part2)
        shape = part1.shape+part2.shape
        dtype = chaospy.poly.typing.dtyping(part1.dtype, part2.dtype)
        return Poly(core_new, part1.dim, shape, dtype)

    raise NotImplementedError


def dot(poly1, poly2):
    """
    Dot product of polynomial vectors.

    Args:
        poly1 (Poly) : left part of product.
        poly2 (Poly) : right part of product.

    Returns:
        (Poly) : product of poly1 and poly2.

    Examples:
        >>> poly = cp.prange(3, 1)
        >>> print(poly)
        [1, q0, q0^2]
        >>> print(cp.dot(poly, np.arange(3)))
        2q0^2+q0
        >>> print(cp.dot(poly, poly))
        q0^4+q0^2+1
    """
    if not isinstance(poly1, Poly) and not isinstance(poly2, Poly):
        return np.dot(poly1, poly2)

    poly1 = Poly(poly1)
    poly2 = Poly(poly2)

    poly = poly1*poly2
    if np.prod(poly1.shape) <= 1 or np.prod(poly2.shape) <= 1:
        return poly
    return chaospy.poly.sum(poly, 0)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
