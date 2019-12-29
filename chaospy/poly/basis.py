"""Create an N-dimensional unit polynomial basis."""
import numpy
import numpoly


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
        (chaospy.poly.ndpoly) : Polynomial array.

    Examples:
        >>> chaospy.basis(4, 4, 2, sort="GR")
        polynomial([q0**4, q0**3*q1, q0**2*q1**2, q0*q1**3, q1**4])
        >>> chaospy.basis([1, 1], [2, 2], sort="GR")
        polynomial([q0*q1, q0**2*q1, q0*q1**2, q0**2*q1**2])
    """
    if stop is None:
        start, stop = 0, start
    dim = max(numpy.asarray(start).size, numpy.asarray(stop).size, dim)
    return numpoly.monomial(
        start=start,
        stop=numpy.array(stop)+1,
        ordering=sort,
        cross_truncation=cross_truncation,
        names=numpoly.symbols("q:%d" % dim, asarray=True),
    )
