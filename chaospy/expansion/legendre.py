import numpy
import chaospy


def legendre(
        order,
        lower=-1,
        upper=1,
        physicist=False,
        normed=False,
        retall=False,
):
    """
    Examples:
        >>> polynomials, norms = chaospy.expansion.legendre(3, retall=True)
        >>> polynomials.round(5)
        polynomial([1.0, q0, q0**2-0.33333, q0**3-0.6*q0])
        >>> norms
        array([1.        , 0.33333333, 0.08888889, 0.02285714])
        >>> chaospy.expansion.legendre(3, physicist=True).round(3)
        polynomial([1.0, 1.5*q0, 2.5*q0**2-0.556, 4.375*q0**3-1.672*q0])
        >>> chaospy.expansion.legendre(3, lower=0, upper=1, normed=True).round(3)
        polynomial([1.0, 3.464*q0-1.732, 13.416*q0**2-13.416*q0+2.236,
                    52.915*q0**3-79.373*q0**2+31.749*q0-2.646])

    """
    multiplier = 1.
    if physicist:
        multiplier = numpy.arange(1, order+1)
        multiplier = (2*multiplier+1)/(multiplier+1)
    _, [polynomials], [norms] = chaospy.recurrence.analytical_stieltjes(
        order, chaospy.Uniform(lower, upper), multiplier=multiplier)
    if normed:
        polynomials = chaospy.true_divide(polynomials, numpy.sqrt(norms))
        norms[:] = 1.
    return (polynomials, norms) if retall else polynomials
