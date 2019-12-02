"""Tools for creating multi-indices"""
from __future__ import division

import itertools
import numpy


def bindex(start, stop=None, dim=1, sort="G", cross_truncation=1.):
    """
    Generator for creating multi-indices.

    Args:
        start (Union[int, numpy.ndarray]):
            The lower order of the indices. If array of int, counts as lower
            bound for each axis.
        stop (Union[int, numpy.ndarray, None]):
            the maximum shape included. If omitted: stop <- start; start <- 0
            If int is provided, set as largest total order. If array of int,
            set as upper bound for each axis.
        dim (int):
            The number of dimensions in the expansion
        sort (str):
            Criteria to sort the indices by.
        cross_truncation (float):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion. Ignored if ``stop`` is a array.

    Returns:
        list:
            Order list of indices.

    Examples:
        >>> bindex(start=2, stop=3, dim=2).T
        array([[0, 1, 2, 0, 1, 2, 3],
               [2, 1, 0, 3, 2, 1, 0]])
        >>> bindex(start=2, stop=[1, 3], dim=2, cross_truncation=0).T
        array([[0, 1, 0, 1, 1],
               [2, 1, 3, 2, 3]])
        >>> bindex(start=[1, 2], stop=[2, 3], dim=2, cross_truncation=0).T
        array([[1, 1, 2, 2],
               [2, 3, 2, 3]])
        >>> bindex(start=1, stop=3, dim=2, cross_truncation=0).T
        array([[0, 1, 0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 2, 3, 3],
               [1, 0, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 3, 2, 3]])
        >>> bindex(start=1, stop=3, dim=2, cross_truncation=1).T
        array([[0, 1, 0, 1, 2, 0, 1, 2, 3],
               [1, 0, 2, 1, 0, 3, 2, 1, 0]])
        >>> bindex(start=1, stop=3, dim=2, cross_truncation=1.5).T
        array([[0, 1, 0, 1, 2, 0, 3],
               [1, 0, 2, 1, 0, 3, 0]])
        >>> bindex(start=1, stop=3, dim=2, cross_truncation=2).T
        array([[0, 1, 0, 2, 0, 3],
               [1, 0, 2, 0, 3, 0]])
        >>> bindex(start=0, stop=1, dim=3).T
        array([[0, 0, 0, 1],
               [0, 0, 1, 0],
               [0, 1, 0, 0]])
        >>> bindex(start=[1, 1], stop=3, dim=2, cross_truncation=0).T
        array([[1, 1, 2, 1, 2, 3, 2, 3, 3],
               [1, 2, 1, 3, 2, 1, 3, 2, 3]])
    """
    if stop is None:
        start, stop = 0, start
    start = numpy.array(start, dtype=int).flatten()
    stop = numpy.array(stop, dtype=int).flatten()
    sort = sort.upper()
    start[start < 0] = 0

    indices = _bindex(start, stop, dim, cross_truncation)
    if "G" in sort:
        indices = indices[numpy.lexsort([numpy.sum(indices, -1)])]

    if "I" in sort:
        indices = indices[::-1]

    if "R" in sort:
        indices = indices[:, ::-1]

    return indices


def _bindex(start, stop, dim=1, cross_truncation=1.):
    # At the beginning the current list of indices just ranges over the
    # last dimension.
    bound = stop.max()+1
    range_ = numpy.arange(bound, dtype=int)
    indices = range_[:, numpy.newaxis]

    for idx in range(dim-1):

        # Repeats the current set of indices.
        # e.g. [0,1,2] -> [0,1,2,0,1,2,...,0,1,2]
        indices = numpy.tile(indices, (bound, 1))

        # Stretches ranges over the new dimension.
        # e.g. [0,1,2] -> [0,0,...,0,1,1,...,1,2,2,...,2]
        front = range_.repeat(len(indices)//bound)[:, numpy.newaxis]

        # Puts them two together.
        indices = numpy.column_stack((front, indices))

        # Truncate at each iteration to ensure memory usage is low enough
        if stop.size == 1 and cross_truncation > 0:
            lhs = numpy.sum(indices**(1/cross_truncation), -1)
            rhs = numpy.max(stop, -1)**(1/cross_truncation)
            indices = indices[lhs <= rhs]
        else:
            indices = indices[numpy.all(indices <= stop, -1)]

    if start.size == 1:
        indices = indices[numpy.sum(indices, -1) >= start.item()]
    else:
        indices = indices[numpy.all(indices >= start, -1)]
    return indices.reshape(-1, dim)
