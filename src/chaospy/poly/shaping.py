"""
Function for changing polynomial array's shape.
"""
import numpy as np

from chaospy.poly.base import Poly


def flatten(vari):
    """
    Flatten a shapeable quantity.

    Args:
        vari (Poly, array_like) : Shapeable input quantity.

    Returns:
        (Poly, array_like) : Same type as `vari` with `len(Q.shape)==1`.

    Examples:
        >>> P = cp.reshape(cp.prange(4), (2,2))
        >>> print(P)
        [[1, q0], [q0^2, q0^3]]
        >>> print(cp.flatten(P))
        [1, q0, q0^2, q0^3]
    """
    if isinstance(vari, Poly):
        shape = int(np.prod(vari.shape))
        return reshape(vari, (shape,))

    return np.array(vari).flatten()


def reshape(vari, shape):
    """
    Reshape the shape of a shapeable quantity.

    Args:
        vari (Poly, array_like) : Shapeable input quantity
        shape (tuple) : The polynomials new shape. Must be compatible with the
                number of elements in `vari`.

    Returns:
        (Poly, array_like) : Same type as `vari`.

    Examples:
        >>> poly = cp.prange(6)
        >>> print(poly)
        [1, q0, q0^2, q0^3, q0^4, q0^5]
        >>> print(cp.reshape(poly, (2,3)))
        [[1, q0, q0^2], [q0^3, q0^4, q0^5]]
    """

    if isinstance(vari, Poly):
        core = vari.A.copy()
        for key in vari.keys:
            core[key] = reshape(core[key], shape)
        out = Poly(core, vari.dim, shape, vari.dtype)
        return out

    return np.asarray(vari).reshape(shape)


def rollaxis(vari, axis, start=0):
    """
    Roll the specified axis backwards, until it lies in a given position.

    Args:
        vari (Poly, array_like) : Input array or polynomial.
        axis (int) : The axis to roll backwards. The positions of the
            other axes do not change realtie to one another.
        start (int, optional) : The axis is rolled until it lies before
            thes position.
    """
    if isinstance(vari, Poly):
        core_old = vari.A.copy()
        core_new = {}
        for key in vari.keys:
            core_new[key] = rollaxis(core_old[key], axis, start)
        return Poly(core_new, vari.dim, None, vari.dtype)

    return np.rollaxis(vari, axis, start)


def swapaxes(vari, ax1, ax2):
    """Interchange two axes of a polynomial."""
    if isinstance(vari, Poly):
        core = vari.A.copy()
        for key in vari.keys:
            core[key] = swapaxes(core[key], ax1, ax2)

        return Poly(core, vari.dim, None, vari.dtype)

    return np.swapaxes(vari, ax1, ax2)


def roll(vari, shift, axis=None):
    """Roll array elements along a given axis."""
    if isinstance(vari, Poly):
        core = vari.A.copy()
        for key in vari.keys:
            core[key] = roll(core[key], shift, axis)
        return Poly(core, vari.dim, None, vari.dtype)

    return np.roll(vari, shift, axis)


def transpose(vari):
    """
    Transpose a shapeable quantety.

    Args:
        vari (Poly, array_like) : Quantety of interest.

    Returns:
        Q (Poly, array_like) : Same type as `vari`.

    Examples:
        >>> P = cp.reshape(cp.prange(4), (2,2))
        >>> print(P)
        [[1, q0], [q0^2, q0^3]]

        >>> print(cp.transpose(P))
        [[1, q0^2], [q0, q0^3]]
    """
    if isinstance(vari, Poly):
        core = vari.A.copy()
        for key in vari.keys:
            core[key] = transpose(core[key])
        return Poly(core, vari.dim, vari.shape[::-1], vari.dtype)

    return np.transpose(vari)


import chaospy as cp

if __name__ == '__main__':
    import doctest
    doctest.testmod()
