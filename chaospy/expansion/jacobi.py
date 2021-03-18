import numpy
import chaospy


def jacobi(
        order,
        alpha,
        beta,
        lower=-1,
        upper=1,
        physicist=False,
        normed=False,
        retall=False,
):
    """
    Jacobi polynomial expansion.

    Examples:
        >>> polynomials, norms = chaospy.expansion.jacobi(4, 0.5, 0.5, retall=True)
        >>> polynomials
        polynomial([1.0, q0, q0**2-0.5, q0**3-0.75*q0, q0**4-q0**2+0.125])
        >>> norms
        array([1.       , 0.5      , 0.125    , 0.03125  , 0.0078125])
        >>> chaospy.expansion.jacobi(3, 0.5, 0.5,  physicist=True).round(4)
        polynomial([1.0, 1.5*q0, 2.5*q0**2-0.8333, 4.375*q0**3-2.1146*q0])
        >>> chaospy.expansion.jacobi(3, 1.5, 0.5, normed=True)
        polynomial([1.0, 2.0*q0, 4.0*q0**2-1.0, 8.0*q0**3-4.0*q0])

    """
    multiplier = 1
    if physicist:
        multiplier = numpy.arange(1, order+1)
        multiplier = ((2*multiplier+alpha+beta-1)*(2*multiplier+alpha+beta)/
                      (2*multiplier*(multiplier+alpha+beta)))
    _, [polynomials], [norms] = chaospy.recurrence.analytical_stieltjes(
        order, chaospy.Beta(alpha, beta, lower=lower, upper=upper), multiplier=multiplier)
    if normed:
        polynomials = chaospy.true_divide(polynomials, numpy.sqrt(norms))
        norms[:] = 1.
    return (polynomials, norms) if retall else polynomials
