"""Hermite orthogonal polynomial expansion."""
import numpy
import chaospy


def hermite(
        order,
        mu=0.,
        sigma=1.,
        physicist=False,
        normed=False,
        retall=False,
):
    """
    Hermite orthogonal polynomial expansion.

    Args:
        order (int):
            The quadrature order.
        mu (float):
            Non-centrality parameter.
        sigma (float):
            Scale parameter.
        physicist (bool):
            Use physicist weights instead of probabilist variant.
        normed (bool):
            If True orthonormal polynomials will be used.
        retall (bool):
            If true return numerical stabilized norms as well. Roughly the same
            as ``cp.E(orth**2, dist)``.

    Returns:
        (numpoly.ndpoly, numpy.ndarray):
            Hermite polynomial expansion. Norms of the orthogonal
            expansion on the form ``E(orth**2, dist)``.

    Examples:
        >>> polynomials, norms = chaospy.expansion.hermite(4, retall=True)
        >>> polynomials
        polynomial([1.0, q0, q0**2-1.0, q0**3-3.0*q0, q0**4-6.0*q0**2+3.0])
        >>> norms
        array([ 1.,  1.,  2.,  6., 24.])
        >>> chaospy.expansion.hermite(3, physicist=True)
        polynomial([1.0, 2.0*q0, 4.0*q0**2-2.0, 8.0*q0**3-12.0*q0])
        >>> chaospy.expansion.hermite(3, sigma=2.5, normed=True).round(3)
        polynomial([1.0, 0.4*q0, 0.113*q0**2-0.707, 0.026*q0**3-0.49*q0])

    """
    multiplier = 2 if physicist else 1
    _, [polynomials], [norms] = chaospy.recurrence.analytical_stieltjes(
        order, chaospy.Normal(mu, sigma), multiplier=multiplier)
    if normed:
        polynomials = chaospy.true_divide(polynomials, numpy.sqrt(norms))
        norms[:] = 1.
    return (polynomials, norms) if retall else polynomials
