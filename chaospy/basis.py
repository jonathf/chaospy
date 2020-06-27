"""Create an N-dimensional unit polynomial basis."""
import logging

import numpy
import numpoly


def basis(start, stop=None, dim=1, cross_truncation=1., sort="G"):
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
        cross_truncation (float):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion.

    Returns:
        (numpoly.ndpoly) : Polynomial array.

    Examples:
        >>> chaospy.basis(2, dim=2)
        polynomial([1, q1, q0, q1**2, q0*q1, q0**2])
        >>> chaospy.basis(2, dim=4, cross_truncation=0)
        polynomial([1, q3, q2, q1, q0, q3**2, q2**2, q1**2, q0**2])
        >>> chaospy.basis(2, 2, dim=2, cross_truncation=numpy.inf)
        polynomial([q1**2, q0**2, q0*q1**2, q0**2*q1, q0**2*q1**2])

    """
    logger = logging.getLogger(__name__)
    logger.warning("chaospy.basis is deprecated; use chaospy.monomial instead")

    if stop is None:
        start, stop = 0, start
    dim = max(numpy.asarray(start).size, numpy.asarray(stop).size, dim)
    out = numpoly.monomial(
        start=start,
        stop=numpy.array(stop)+1,
        reverse="R" not in sort.upper(),
        graded="G" in sort.upper(),
        cross_truncation=cross_truncation,
        names=numpoly.symbols("q:%d" % dim, asarray=True),
    )
    if "I" in sort.upper():
        out = out[::-1]
    return out
