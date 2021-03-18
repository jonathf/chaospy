import numpy
import chaospy


def gegenbauer(
        order,
        alpha,
        lower=-1,
        upper=1,
        physicist=False,
        normed=False,
        retall=False,
):
    """
    Gegenbauer polynomials.

    Args:
        order (int):
            The polynomial order.
        alpha (float):
            Gegenbauer shape parameter.
        lower (float):
            Lower bound for the integration interval.
        upper (float):
            Upper bound for the integration interval.
        physicist (bool):
            Use physicist weights instead of probabilist.

    Examples:
        >>> polynomials, norms = chaospy.expansion.gegenbauer(4, 1, retall=True)
        >>> polynomials
        polynomial([1.0, q0, q0**2-0.25, q0**3-0.5*q0, q0**4-0.75*q0**2+0.0625])
        >>> norms
        array([1.        , 0.25      , 0.0625    , 0.015625  , 0.00390625])
        >>> chaospy.expansion.gegenbauer(3, 1, physicist=True)
        polynomial([1.0, 2.0*q0, 4.0*q0**2-0.5, 8.0*q0**3-2.0*q0])
        >>> chaospy.expansion.gegenbauer(3, 1, lower=0.5, upper=1.5, normed=True).round(3)
        polynomial([1.0, 4.0*q0-4.0, 16.0*q0**2-32.0*q0+15.0,
                    64.0*q0**3-192.0*q0**2+184.0*q0-56.0])

    """
    multiplier = 1
    if physicist:
        multiplier = numpy.arange(1, order+1)
        multiplier = 2*(multiplier+alpha-1)/multiplier
    _, [polynomials], [norms] = chaospy.recurrence.analytical_stieltjes(
        order, chaospy.Beta(alpha+0.5, alpha+0.5, lower, upper), multiplier=multiplier)
    if normed:
        polynomials = chaospy.true_divide(polynomials, numpy.sqrt(norms))
        norms[:] = 1.
    return (polynomials, norms) if retall else polynomials
