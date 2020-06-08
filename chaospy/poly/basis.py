"""Create an N-dimensional unit polynomial basis."""
import logging

import numpy
import numpoly


def basis(start, stop=None, dim=1, graded=True, reverse=True, cross_truncation=1., sort=None):
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
        graded (bool):
            Graded sorting, meaning the indices are always sorted by the index
            sum. E.g. ``q0**2*q1**2*q2**2`` has an exponent sum of 6, and will
            therefore be consider larger than both ``q0**2*q1*q2``,
            ``q0*q1**2*q2`` and ``q0*q1*q2**2``, which all have exponent sum of
            5.
        reverse (bool):
            Reverse lexicographical sorting meaning that ``q0*q1**3`` is
            considered bigger than ``q0**3*q1``, instead of the opposite.
        cross_truncation (float):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion.

    Returns:
        (chaospy.poly.ndpoly) : Polynomial array.

    Examples:
        >>> chaospy.basis(2, dim=2)
        polynomial([1, q1, q0, q1**2, q0*q1, q0**2])
        >>> chaospy.basis(2, dim=4, cross_truncation=0)
        polynomial([1, q3, q2, q1, q0, q3**2, q2**2, q1**2, q0**2])
        >>> chaospy.basis(2, 2, dim=2, cross_truncation=numpy.inf)
        polynomial([q1**2, q0**2, q0*q1**2, q0**2*q1, q0**2*q1**2])

    """
    logger = logging.getLogger(__name__)
    if sort is not None:
        logger.warning("deprecation warning: 'sort' argument is deprecated; "
                       "use 'graded' and/or 'reverse' instead")
        graded = "G" in sort.upper()
        reverse = "R" not in sort.upper()
        inverse = "I" in sort.upper()
        out = basis(start, stop, dim, graded, reverse, cross_truncation)
        if inverse:
            out = out[::-1]
        return out

    if stop is None:
        start, stop = 0, start
    dim = max(numpy.asarray(start).size, numpy.asarray(stop).size, dim)
    return numpoly.monomial(
        start=start,
        stop=numpy.array(stop)+1,
        reverse=reverse,
        graded=graded,
        cross_truncation=cross_truncation,
        names=numpoly.symbols("q:%d" % dim, asarray=True),
    )
