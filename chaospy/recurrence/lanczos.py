"""Discretized Lanczos' method."""
import logging
import numpy
import chaospy


def lanczos(
        order,
        dist,
        rule="clenshaw_curtis",
        tolerance=1e-12,
        scaling=3,
        n_max=1e4,
):
    """
    Discretized Lanczos' method.

    Iterative increase the quadrature order until the norms converges.

    Args:
        order (int):
            The order create recurrence coefficients for.
        dist (chaospy.Distribution):
            The distribution to create recurrence coefficients with respect to.
        rule (str):
            The rule to use to create quadrature nodes and weights from.
        tolerance (float):
            The allowed relative error in norm between two quadrature orders
            before method assumes convergence.
        scaling (float):
            A multiplier the adaptive order increases with for each step
            quadrature order is not converged. Use 0 to indicate unit
            increments.
        n_max (int):
            The allowed number of quadrature points to use in approximation.

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray):
            coefficients:
                The recurrence coefficients created using the discretized
                Stieltjes' method, with ``shape == (2, D, order+1)``.

    Examples:
        >>> dist = chaospy.J(chaospy.Beta(3, 6), chaospy.Normal())
        >>> alpha, beta = chaospy.lanczos(3, dist)
        >>> alpha.round(5)
        array([[ 0.33333,  0.39394,  0.42657,  0.44615],
               [-0.     , -0.     , -0.     , -0.     ]])
        >>> beta.round(5)
        array([[1.     , 0.02222, 0.03471, 0.04227],
               [1.     , 1.     , 2.     , 3.     ]])

    Notes:
        The script is adapted from the routine RKPW in W.B. Gragg and W.J.
        Harrod, "The numerically stable reconstruction of Jacobi matrices from
        spectral data", Numer. Math. 44 (1984), 317-335.

    """
    logger = logging.getLogger(__name__)
    if len(dist) > 1:
        assert not dist.stochastic_dependent
        coeffs = zip(*[lanczos(
                order,
                dist_,
                rule=rule,
                tolerance=tolerance,
                scaling=scaling)
        for dist_ in dist])
        coeffs = numpy.vstack(list(coeffs)).reshape((2, len(dist), order+1))
        return coeffs

    order_ = (2*order-1.)/scaling
    beta = beta_old = numpy.nan
    coeffs = numpy.ones((2, order+1))

    while not numpy.all(numpy.abs(beta-beta_old) < tolerance):

        order_ = max(order_*scaling, order_+1)
        if order_ > n_max:
            logger.warning("number of nodes exceeded; stopping with errors:\n%s",
                           ", ".join([numpy.format_float_scientific(val, 1)
                                      for val in numpy.abs(beta-beta_old)/beta]))
            break

        [abscissas], weights = chaospy.generate_quadrature(
            int(order_), dist, rule=rule, segments=0)
        alpha = abscissas[:order+1].copy()
        beta, beta_old = numpy.eye(order+1)[0]*weights[0], beta

        for idx in range(1, len(weights)):

            gamma = 1
            increment_new = 0

            for idy in range(min(idx, order+1)):

                increment = increment_new
                beta_new = gamma*(beta[idy]+weights[idx])
                sigma = 1-gamma
                gamma = numpy.where(
                    beta[idy] <= -weights[idx], 1,
                    beta[idy]/(beta[idy]+weights[idx]))
                increment_new = (
                    (1-gamma)*(alpha[idy]-abscissas[idx])-gamma*increment)
                weights[idx] = numpy.where(
                    gamma >= 1, sigma*beta[idy], increment_new**2/(1-gamma))

                alpha[idy] -= increment_new-increment
                beta[idy] = beta_new

    return numpy.asfarray([alpha.ravel(), beta.ravel()])
