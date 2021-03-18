import numpy
import chaospy


def laguerre(
        order,
        alpha=0.,
        physicist=False,
        normed=False,
        retall=False,
):
    """
    Examples:
        >>> polynomials, norms = chaospy.expansion.laguerre(3, retall=True)
        >>> polynomials
        polynomial([1.0, q0-1.0, q0**2-4.0*q0+2.0, q0**3-9.0*q0**2+18.0*q0-6.0])
        >>> norms
        array([ 1.,  1.,  4., 36.])
        >>> chaospy.expansion.laguerre(3, physicist=True).round(5)
        polynomial([1.0, -q0+1.0, 0.5*q0**2-2.0*q0+2.0,
                    -0.16667*q0**3+1.5*q0**2-5.33333*q0+4.66667])
        >>> chaospy.expansion.laguerre(3, alpha=2, normed=True).round(3)
        polynomial([1.0, 0.577*q0-1.732, 0.204*q0**2-1.633*q0+2.449,
                    0.053*q0**3-0.791*q0**2+3.162*q0-3.162])

    """
    multiplier = -1./numpy.arange(1, order+1) if physicist else 1.
    _, [polynomials], [norms] = chaospy.recurrence.analytical_stieltjes(
        order, chaospy.Gamma(alpha+1), multiplier=multiplier)
    if normed:
        polynomials = chaospy.true_divide(polynomials, numpy.sqrt(norms))
        norms[:] = 1.
    return (polynomials, norms) if retall else polynomials
