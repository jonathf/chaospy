"""Constructor to create a range of polynomials where the exponent vary."""
import logging
import numpoly


def prange(N=1, dim=1):
    """
    Constructor to create a range of polynomials where the exponent vary.

    Args:
        N (int):
            Number of polynomials in the array.
        dim (int):
            The dimension the polynomial should span.

    Returns:
        (numpoly.ndpoly):
            A polynomial array of length N containing simple polynomials with
            increasing exponent.

    Examples:
        >>> chaospy.prange(4)
        polynomial([1, q0, q0**2, q0**3])
        >>> chaospy.prange(4, dim=3)
        polynomial([1, q2, q2**2, q2**3])
    """
    logger = logging.getLogger(__name__)
    logger.warning(
        "chaospy.prange is deprecated; use chaospy.monomial instead")
    out = numpoly.monomial(N)
    if dim > 1:
        out = numpoly.call(
            out, kwargs={out.names[0]: numpoly.variable(dim)[-1]})
    return out
