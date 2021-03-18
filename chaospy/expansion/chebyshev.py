"""Chebyshev polynomials of the first kind."""
import numpy
import chaospy


def chebyshev_1(
        order,
        lower=-1,
        upper=1,
        physicist=False,
        normed=False,
        retall=False,
):
    """
    Chebyshev polynomials of the first kind.

    Args:
        order (int):
            The polynomial order.
        lower (float):
            Lower bound for the integration interval.
        upper (float):
            Upper bound for the integration interval.
        physicist (bool):
            Use physicist weights instead of probabilist.

    Returns:
        (numpoly.ndpoly, numpy.ndarray):
            Chebyshev polynomial expansion. Norms of the orthogonal
            expansion on the form ``E(orth**2, dist)``.

    Examples:
        >>> polynomials, norms = chaospy.expansion.chebyshev_1(4, retall=True)
        >>> polynomials
        polynomial([1.0, q0, q0**2-0.5, q0**3-0.75*q0, q0**4-q0**2+0.125])
        >>> norms
        array([1.       , 0.5      , 0.125    , 0.03125  , 0.0078125])
        >>> chaospy.expansion.chebyshev_1(3, physicist=True)
        polynomial([1.0, q0, 2.0*q0**2-1.0, 4.0*q0**3-2.5*q0])
        >>> chaospy.expansion.chebyshev_1(3, lower=0.5, upper=1.5, normed=True).round(3)
        polynomial([1.0, 2.828*q0-2.828, 11.314*q0**2-22.627*q0+9.899,
                    45.255*q0**3-135.765*q0**2+127.279*q0-36.77])

    """
    multiplier = 1+numpy.arange(order).astype(bool) if physicist else 1
    _, [polynomials], [norms] = chaospy.recurrence.analytical_stieltjes(
        order, chaospy.Beta(0.5, 0.5, lower, upper), multiplier=multiplier)
    if normed:
        polynomials = chaospy.true_divide(polynomials, numpy.sqrt(norms))
        norms[:] = 1.
    return (polynomials, norms) if retall else polynomials


def chebyshev_2(
        order,
        lower=-1,
        upper=1,
        physicist=False,
        normed=False,
        retall=False,
):
    """
    Chebyshev polynomials of the second kind.

    Args:
        order (int):
            The quadrature order.
        lower (float):
            Lower bound for the integration interval.
        upper (float):
            Upper bound for the integration interval.
        physicist (bool):
            Use physicist weights instead of probabilist.

    Returns:
        (numpoly.ndpoly, numpy.ndarray):
            Chebyshev polynomial expansion. Norms of the orthogonal
            expansion on the form ``E(orth**2, dist)``.

    Examples:
        >>> polynomials, norms = chaospy.expansion.chebyshev_2(4, retall=True)
        >>> polynomials
        polynomial([1.0, q0, q0**2-0.25, q0**3-0.5*q0, q0**4-0.75*q0**2+0.0625])
        >>> norms
        array([1.        , 0.25      , 0.0625    , 0.015625  , 0.00390625])
        >>> chaospy.expansion.chebyshev_2(3, physicist=True)
        polynomial([1.0, 2.0*q0, 4.0*q0**2-0.5, 8.0*q0**3-2.0*q0])
        >>> chaospy.expansion.chebyshev_2(3, lower=0.5, upper=1.5, normed=True).round(3)
        polynomial([1.0, 4.0*q0-4.0, 16.0*q0**2-32.0*q0+15.0,
                    64.0*q0**3-192.0*q0**2+184.0*q0-56.0])

    """
    multiplier = 2 if physicist else 1
    _, [polynomials], [norms] = chaospy.recurrence.analytical_stieltjes(
        order, chaospy.Beta(1.5, 1.5, lower, upper), multiplier=multiplier)
    if normed:
        polynomials= chaospy.true_divide(polynomials, numpy.sqrt(norms))
        norms[:] = 1.
    return (polynomials, norms) if retall else polynomials
