"""
Function that overlaps with the numpy library.
"""
import numpy as np

import chaospy as cp
import chaospy.poly
from chaospy.poly.base import Poly


def sum(vari, axis=None): # pylint: disable=redefined-builtin
    """
    Sum the components of a shapeable quantity along a given axis.

    Args:
        vari (Poly, array_like) : Input data.
        axis (int, optional) : Axis over which the sum is taken. By default
                `axis` is None, and all elements are summed.

    Returns:
        (Poly, array_like) : Polynomial array with same shape as `vari`,
                with the specified axis removed. If `vari` is an 0-d array, or
                `axis` is None, a (non-iterable) component is returned.

    Examples:
        >>> vari = cp.prange(3)
        >>> print(vari)
        [1, q0, q0^2]
        >>> print(cp.sum(vari))
        q0^2+q0+1
    """
    if isinstance(vari, Poly):

        core = vari.A.copy()
        for key in vari.keys:
            core[key] = sum(core[key], axis)

        return Poly(core, vari.dim, None, vari.dtype)

    return np.sum(vari, axis)


def cumsum(vari, axis=None):
    """
    Cumulative sum the components of a shapeable quantity along a given axis.

    Args:
        vari (Poly, array_like) : Input data.
        axis (int, optional) : Axis over which the sum is taken. By default
                `axis` is None, and all elements are summed.

    Returns:
        (Poly, array_like) : Polynomial array with same shape as `vari`.

    Examples:
        >>> poly = cp.prange(3)
        >>> print(poly)
        [1, q0, q0^2]
        >>> print(cp.cumsum(poly))
        [1, q0+1, q0^2+q0+1]
    """
    if isinstance(vari, Poly):
        core = vari.A.copy()
        for key, val in core.items():
            core[key] = cumsum(val, axis)
        return Poly(core, vari.dim, None, vari.dtype)

    return np.cumsum(vari, axis)


def prod(vari, axis=None):
    """
    Product of the components of a shapeable quantity along a given axis.

    Args:
        vari (Poly, array_like) : Input data.
        axis (int, optional) : Axis over which the sum is taken. By default
                `axis` is None, and all elements are summed.

    Returns:
        (Poly, array_like) : Polynomial array with same shape as `vari`,
                with the specified axis removed. If `vari` is an 0-d array, or
                `axis` is None, a (non-iterable) component is returned.

    Examples:
        >>> vari = cp.prange(3)
        >>> print(vari)
        [1, q0, q0^2]
        >>> print(cp.prod(vari))
        q0^3
    """
    if isinstance(vari, Poly):
        if axis is None:
            vari = chaospy.poly.shaping.flatten(vari)
            axis = 0

        vari = chaospy.poly.shaping.rollaxis(vari, axis)
        out = vari[0]
        for poly in vari[1:]:
            out = out*poly
        return out

    return np.prod(vari, axis)


def cumprod(vari, axis=None):
    """
    Perform the cumulative product of a shapeable quantity over a given axis.

    Args:
        vari (Poly, array_like) : Input data.
        axis (int, optional) : Axis over which the product is taken.  By
                default, the product of all elements is calculated.

    Returns:
        (Poly) : An array shaped as `vari` but with the specified axis removed.

    Examples:
        >>> vari = cp.prange(4)
        >>> print(vari)
        [1, q0, q0^2, q0^3]
        >>> print(cp.cumprod(vari))
        [1, q0, q0^3, q0^6]
    """
    if isinstance(vari, Poly):
        if np.prod(vari.shape) == 1:
            return vari.copy()
        if axis is None:
            vari = chaospy.poly.shaping.flatten(vari)
            axis = 0

        vari = chaospy.poly.shaping.rollaxis(vari, axis)
        out = [vari[0]]

        for poly in vari[1:]:
            out.append(out[-1]*poly)
        return Poly(out, vari.dim, vari.shape, vari.dtype)

    return np.cumprod(vari, axis)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
