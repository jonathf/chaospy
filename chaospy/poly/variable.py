"""Simple constructor to create single variables to create polynomials."""
import numpoly


def variable(dims=1):
    """
    Simple constructor to create single variables to create polynomials.

    Args:
        dims (int):
            Number of dimensions in the array.

    Returns:
        (chaospy.poly.polynomial):
            Polynomial array with unit components in each dimension.

    Examples:
        >>> chaospy.variable()
        polynomial(q0)
        >>> chaospy.variable(3)
        polynomial([q0, q1, q2])
    """
    return numpoly.symbols("q:%d" % dims)
