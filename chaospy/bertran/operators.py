"""
Basic tools for Bertran index manipulation.
"""
import functools
import itertools

import numpy
from scipy.special import comb

import chaospy.bertran
from .indices import bindex

_ADD_CACHE = {}
_MULTI_INDEX_CACHE = {}


def add(idxi, idxj, dim):
    """
    Bertran addition.

    Args:
        idxi (Tuple):
            Index in integer notation
        idxj (Tuple):
            Index in integer notation
        dim (int):
            The number of dimensions of the expansion.

    Examples:
        >>> chaospy.bertran.add(3, 3, 1)
        6
        >>> chaospy.bertran.add(3, 3, 2)
        10
    """
    key = idxi, idxj, dim
    if key in _ADD_CACHE:
        return _ADD_CACHE[key]

    idxi = multi_index(idxi, dim)
    idxj = multi_index(idxj, dim)
    out = single_index(tuple(i+j for i,j in zip(idxi, idxj)))

    _ADD_CACHE[key] = out
    return out


def terms(order, dim):
    """
    Count the number of polynomials in an expansion.

    Args:
        order (int):
            The upper order for the expansion.
        dim (int):
            The number of dimensions of the expansion.

    Returns:
        The number of terms in an expansion of upper order ``order`` and number
        of dimensions ``dim``.
    """
    return int(comb(order+dim, dim, exact=True))


def multi_index(idx, dim):
    """
    Single to multi-index using graded reverse lexicographical notation.

    Args:
        idx (int):
            Index in integer notation
        dim (int):
            The number of dimensions in the multi-index notation

    Returns (Tuple):
        Multi-index of ``idx`` with ``len(out) == dim``.

    Examples:
        >>> for idx in range(5):
        ...     print(chaospy.bertran.multi_index(idx, 3))
        (0, 0, 0)
        (1, 0, 0)
        (0, 1, 0)
        (0, 0, 1)
        (2, 0, 0)

    See Also:
        :func:`single_index`
    """
    key = idx, dim
    if key in _MULTI_INDEX_CACHE:
        return _MULTI_INDEX_CACHE[key]

    if not dim:
        out = ()

    elif idx == 0:
        out = (0, )*dim

    else:
        idxn = idxm = 0
        while terms(idxn, dim) <= idx:
            idxn += 1
        idx -= terms(idxn-1, dim)

        if idx == 0:
            out = (idxn,) + (0,)*(dim-1)
        else:
            while terms(idxm, dim-1) <= idx:
                idxm += 1
            out = (int(idxn-idxm),) + multi_index(idx, dim-1)

    _MULTI_INDEX_CACHE[key] = out
    return out


def single_index(idxm):
    """
    Multi-index to single integer notation.

    Uses graded reverse lexicographical notation.

    Parameters
    ----------
    idxm : numpy.ndarray
        Index in multi-index notation

    Returns
    -------
    idx : int
        Integer index of `idxm`

    Examples
    --------
    >>> for idx in range(3):
    ...     print(chaospy.bertran.single_index(numpy.eye(3)[idx]))
    1
    2
    3
    """
    if -1 in idxm:
        return 0
    order = int(sum(idxm))
    dim = len(idxm)
    if order == 0:
        return 0
    return terms(order-1, dim) + single_index(idxm[1:])


def rank(idx, dim):
    """Calculate the index rank according to Bertran's notation."""
    idxm = multi_index(idx, dim)
    out = 0
    while idxm[-1:] == (0,):
        out += 1
        idxm = idxm[:-1]
    return out


def parent(idx, dim, axis=None):
    """
    Parent node according to Bertran's notation.

    Parameters
    ----------
    idx : int
        Index of the child node.
    dim : int
        Dimensionality of the problem.
    axis : int
        Assume axis direction.

    Returns
    -------
    out : int
        Index of parent node with `j<=i`, and `j==i` iff `i==0`.
    axis : int
        Dimension direction the parent was found.
    """
    idxm = multi_index(idx, dim)
    if axis is None:
        axis = dim - numpy.argmin(1*(numpy.array(idxm)[::-1] == 0))-1

    if not idx:
        return idx, axis

    if idxm[axis] == 0:
        idxi = parent(parent(idx, dim)[0], dim)[0]
        while child(idxi+1, dim, axis) < idx:
            idxi += 1
        return idxi, axis

    out = numpy.array(idxm) - 1*(numpy.eye(dim)[axis])
    return single_index(out), axis


def child(idx, dim, axis):
    """
    Child node according to Bertran's notation.

    Parameters
    ----------
    idx : int
        Index of the parent node.
    dim : int
        Dimensionality of the problem.
    axis : int
        Dimension direction to define a child.
        Must have `0<=axis<dim`

    Returns
    -------
    out : int
        Index of child node with `out > idx`.

    Examples
    --------
    >>> chaospy.bertran.child(4, 1, 0)
    5
    >>> chaospy.bertran.child(4, 2, 1)
    8
    """
    idxm = multi_index(idx, dim)
    out = numpy.array(idxm) + 1*(numpy.eye(len(idxm))[axis])
    return single_index(out)


def olindex(order, dim):
    """
    Create an lexiographical sorted basis for a given order.

    Examples
    --------
    >>> chaospy.bertran.olindex(3, 2)
    array([[0, 3],
           [1, 2],
           [2, 1],
           [3, 0]])
    """
    idxm = [0]*dim
    out = []

    def _olindex(idx):
        """Recursive backend for olindex."""
        if numpy.sum(idxm) == order:
            out.append(idxm[:])
            return

        if idx == dim:
            return

        idxm_sum = numpy.sum(idxm)
        idx_saved = idxm[idx]

        for idxi in range(order - numpy.sum(idxm) + 1):

            idxm[idx] = idxi

            if idxm_sum < order:
                _olindex(idx+1)

            else:
                break
        idxm[idx] = idx_saved

    _olindex(0)
    return numpy.array(out)


def olindices(order, dim):
    """
    Create an lexiographical sorted basis for a given order.

    Examples:
        >>> chaospy.bertran.olindices(2, 2)
        array([[0, 0],
               [0, 1],
               [1, 0],
               [0, 2],
               [1, 1],
               [2, 0]])
    """
    indices = [olindex(o, dim) for o in range(order+1)]
    indices = numpy.vstack(indices)
    return indices
