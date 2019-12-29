"""Adjust the dimensions of a polynomial."""
import numpoly


def setdim(poly, dim=None):
    """
    Adjust the dimensions of a polynomial.

    Output the results into ndpoly object

    Args:
        poly (chaospy.poly.ndpoly):
            Input polynomial
        dim (int):
            The dimensions of the output polynomial. If omitted, increase
            polynomial with one dimension. If the new dim is smaller then
            `poly`'s dimensions, variables with cut components are all cut.

    Examples:
        >>> x, y = chaospy.variable(2)
        >>> poly = x*x-x*y
        >>> chaospy.setdim(poly, 1)
        polynomial(q0**2)
        >>> chaospy.setdim(poly, 3)
        polynomial(q0**2-q0*q1)
        >>> chaospy.setdim(poly, 3).names
        ('q0', 'q1', 'q2')
    """
    poly = numpoly.polynomial(poly)
    indices = [int(name[1:]) for name in poly.names]
    dim = max(indices)+2 if dim is None else dim
    poly = poly(**{("q%d" % index): 0 for index in indices if index >= dim})
    _, poly = numpoly.align_indeterminants(numpoly.symbols("q:%d" % dim), poly)
    return poly
