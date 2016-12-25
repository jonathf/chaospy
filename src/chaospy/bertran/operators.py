"""
Basic tools for Bertran index manipulation.
"""
import functools

import numpy
import scipy.misc

import chaospy.bertran


def add(idxi, idxj, dim):
    """
    Bertran addition.

    Example
    -------
    >>> print(chaospy.bertran.add(3, 3, 1))
    6
    >>> print(chaospy.bertran.add(3, 3, 2))
    10
    """
    idxm = numpy.array(multi_index(idxi, dim))
    idxn = numpy.array(multi_index(idxj, dim))
    out = single_index(idxm + idxn)
    return out


def terms(order, dim):
    """
    Count the number of polynomials in an expansion.

    Parameters
    ----------
    order : int
        The upper order for the expansion.
    dim : int
        The number of dimensions of the expansion.

    Returns
    -------
    N : int
        The number of terms in an expansion of upper order `M` and
        number of dimensions `dim`.
    """
    return int(scipy.misc.comb(order+dim, dim, 1))


def multi_index(idx, dim):
    """
    Single to multi-index using graded reverse lexicographical notation.

    Parameters
    ----------
    idx : int
        Index in interger notation
    dim : int
        The number of dimensions in the multi-index notation

    Returns
    -------
    out : tuple
        Multi-index of `idx` with `len(out)=dim`

    Examples
    --------
    >>> for idx in range(5):
    ...     print(chaospy.bertran.multi_index(idx, 3))
    (0, 0, 0)
    (1, 0, 0)
    (0, 1, 0)
    (0, 0, 1)
    (2, 0, 0)

    See Also
    --------
    single_index
    """
    def _rec(idx, dim):
        idxn = idxm = 0
        if not dim:
            return ()

        if idx == 0:
            return (0, )*dim

        while terms(idxn, dim) <= idx:
            idxn += 1
        idx -= terms(idxn-1, dim)

        if idx == 0:
            return (idxn,) + (0,)*(dim-1)
        while terms(idxm, dim-1) <= idx:
            idxm += 1

        return (int(idxn-idxm),) + _rec(idx, dim-1)

    return _rec(idx, dim)


def bindex(start, stop=None, dim=1, sort="G"):
    """
    Generator for creating multi-indices.

    Parameters
    ----------
    start : int
        The lower order of the indices
    stop : int, optional
        the maximum shape included. If omitted:
        stop <- start; start <- 0
        If int is provided, set as largest total order.
        If array of int, set as largest order along each axis.
    dim : int
        The number of dimensions in the expansion

    Returns
    -------
    indices : list
        Grevlex order list of indices.

    Examples
    --------
    >>> print(chaospy.bertran.bindex(0, 1, 3))
    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    >>> print(chaospy.bertran.bindex(2, 3, 2))
    [(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
    """
    if stop is None:
        start, stop = 0, start
    sort = sort.upper()

    local, total = [], []
    for idx in range(
            single_index([start] + [0]*(dim-1)),
            single_index([0]*(dim-1) + [start])+1):
        idxm = multi_index(idx, dim)
        local.append(list(idxm))
        total.append(idxm)

    for _ in range(start, stop):

        local_, local = local, []
        for idxm in local_:

            idxi = 0
            for idxi in range(len(idxm)):
                if sum(idxm[idxi+1:]) == 0:
                    break

            for idxj in range(idxi, len(idxm)):
                idxm[idxj] += 1
                local.append(idxm[:])
                total.append(tuple(idxm))
                idxm[idxj] -= 1

    if "G" in sort:
        total = sorted(total, key=sum)

    else:
        def cmp_(idxi, idxj):
            """Old style compare method."""
            if not numpy.any(idxi):
                return 0
            if idxi[0] == idxj[0]:
                return cmp(idxi[:-1], idxj[:-1])
            return (idxi[-1] > idxj[-1]) - (idxi[-1] < idxj[-1])
        key = functools.cmp_to_key(cmp_)
        total = sorted(total, key=key)

    if "I" in sort:
        total = total[::-1]

    if "R" in sort:
        total = [idx[::-1] for idx in total]

    return total


def single_index(idxm):
    """
    Multi-index to single integer notation.

    Uses graded reverse lexicographical notation.

    Parameters
    ----------
    idxm : array_like
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
    >>> print(chaospy.bertran.child(4, 1, 0))
    5
    >>> print(chaospy.bertran.child(4, 2, 1))
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
