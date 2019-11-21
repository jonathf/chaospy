"""Adjust the dimensions of a polynomial."""
import numpoly


def setdim(P, dim=None):
    """
    Adjust the dimensions of a polynomial.

    Output the results into ndpoly object

    Args:
        P (chaospy.poly.ndpoly) : Input polynomial
        dim (int) : The dimensions of the output polynomial. If omitted,
                increase polynomial with one dimension. If the new dim is
                smaller then P's dimensions, variables with cut components are
                all cut.

    Examples:
        >>> x, y = chaospy.variable(2)
        >>> P = x*x-x*y
        >>> print(chaospy.setdim(P, 1))
        q0**2
        >>> print(chaospy.setdim(P, 3))
        q0**2-q0*q1
        >>> print(chaospy.setdim(P, 3).names)
        ('q0', 'q1', 'q2')
    """
    P = numpoly.polynomial(P)
    indices = [int(name[1:]) for name in P.names]
    dim = max(indices)+2 if dim is None else dim
    P = P(**{("q%d" % index): 0 for index in indices if index >= dim})
    _, P = numpoly.align_indeterminants(numpoly.symbols("q:%d" % dim), P)
    return P
