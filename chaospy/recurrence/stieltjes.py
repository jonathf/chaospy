"""Discretized Stieltjes' method."""
import numpy
import numpoly
import chaospy


def stieltjes(
        order,
        dist,
        rule=None,
        tolerance=1e-16,
        scaling=3,
        n_max=5000,
):
    """
    Stieltjes' method.

    Tries to get recurrence coefficients using the distributions own
    TTR-method, but will fall back to a iterative method if missing.

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
            polynomials:
                The orthogonal polynomial expansion created as a by-product of
                the algorithm.
            norms:
                The norm of each orthogonal polynomial. Roughly equivalent to
                ``chaospy.E(polynomials**2, dist)``, but more numerically
                stable than most alternatives.

    Examples:
        >>> dist = chaospy.J(chaospy.Uniform(0, 1), chaospy.Beta(3, 4))
        >>> (alpha, beta), orth, norms = chaospy.stieltjes(2, dist)
        >>> alpha.round(5)
        array([[0.5    , 0.5    , 0.5    ],
               [0.42857, 0.46032, 0.47475]])
        >>> beta.round(5)
        array([[1.     , 0.08333, 0.06667],
               [1.     , 0.03061, 0.04321]])
        >>> orth[:, 2].round(5)
        polynomial([q0**2-q0+0.16667, q1**2-0.88889*q1+0.16667])
        >>> norms.round(5)
        array([[1.     , 0.08333, 0.00556],
               [1.     , 0.03061, 0.00132]])

    """
    try:
        return analytical_stieltjes(order=order, dist=dist)
    except NotImplementedError:
        return discretized_stieltjes(
            order=order,
            dist=dist,
            rule=rule,
            tolerance=tolerance,
            scaling=scaling,
            n_max=n_max,
        )


def discretized_stieltjes(
        order,
        dist,
        rule=None,
        tolerance=1e-16,
        scaling=3,
        n_max=5000,
):
    """
    Discretized Stieltjes' method.

    Examples:
        >>> dist = chaospy.J(chaospy.Uniform(0, 1), chaospy.Beta(3, 4))
        >>> (alpha, beta), orth, norms = chaospy.discretized_stieltjes(2, dist)
        >>> alpha.round(5)
        array([[0.5    , 0.5    , 0.5    ],
               [0.42857, 0.46032, 0.47475]])
        >>> beta.round(5)
        array([[1.     , 0.08333, 0.06667],
               [1.     , 0.03061, 0.04321]])
        >>> orth[:, 2].round(5)
        polynomial([q0**2-q0+0.16667, q1**2-0.88889*q1+0.16667])
        >>> norms.round(5)
        array([[1.     , 0.08333, 0.00556],
               [1.     , 0.03061, 0.00132]])

    """
    if len(dist) > 1:
        assert not dist.stochastic_dependent
        coeffs, orths, norms = zip(*[discretized_stieltjes(
                order,
                dist_,
                rule=rule,
                tolerance=tolerance,
                scaling=scaling)
        for dist_ in dist])
        coeffs = numpy.dstack(coeffs).reshape(2, len(dist), order+1)
        variables = list(numpoly.variable(len(dist)))
        orths = [orths[idx](q0=variables[idx]) for idx in range(len(dist))]
        orths = numpoly.polynomial(orths).reshape(len(dist), order+1)
        norms = numpy.asfarray(norms).reshape(len(dist), order+1)
        return coeffs, orths, norms

    if rule is None:
        rule = "discrete" if dist.interpret_as_integer else "clenshaw_curtis"
    order_ = (2*order-1.)/scaling
    beta = beta_old = numpy.nan
    var = numpoly.variable()
    orths = [numpoly.polynomial(0.), numpoly.polynomial(1.)]+[None]*order
    norms = numpy.ones(order+2)
    coeffs = numpy.ones((2, order+1))

    while not numpy.all(numpy.abs(coeffs[1]-beta_old) < tolerance):

        beta_old = coeffs[1].copy()
        order_ = max(order_*scaling, order_+1)
        if order_ > n_max:
            break

        [abscissas], weights = chaospy.generate_quadrature(
            int(order_), dist, rule=rule, segments=0)
        inner = numpy.sum(abscissas*weights)
        for idx in range(order):
            coeffs[0, idx] = inner/norms[idx+1]
            coeffs[1, idx] = norms[idx+1]/norms[idx]
            orths[idx+2] = ((var-coeffs[0, idx])*orths[idx+1]-
                            orths[idx]*coeffs[1, idx])
            norms[idx+2] = numpy.sum(orths[idx+2](abscissas)**2*weights)
            inner = numpy.sum(abscissas*orths[idx+2](abscissas)**2*weights)
        coeffs[:, order] = (inner/norms[-1], norms[-1]/norms[-2])

    coeffs = coeffs.reshape(2, 1, order+1)
    orths = numpoly.polynomial(orths[1:]).reshape(1, order+1)
    norms = numpy.array(norms[1:]).reshape(1, order+1)
    return coeffs, orths, norms


def analytical_stieltjes(order, dist, multiplier=1):
    """Analytical Stieltjes' method"""
    dimensions = len(dist)
    mom_order = numpy.arange(order+1).repeat(dimensions)
    mom_order = mom_order.reshape(order+1, dimensions).T
    coeffs = dist.ttr(mom_order)
    coeffs[1, :, 0] = 1.
    orders = numpy.arange(order, dtype=int)
    multiplier, orders = numpy.broadcast_arrays(multiplier, orders)

    var = numpoly.variable(dimensions)
    orth = [numpy.zeros(dimensions), numpy.ones(dimensions)]
    for order_, multiplier_ in zip(orders, multiplier):
        orth.append(
            multiplier_*((var-coeffs[0, :, order_])*orth[-1]-coeffs[1, :, order_]*orth[-2]))
    orth = numpoly.polynomial(orth[1:]).T
    norms = numpy.cumprod(coeffs[1], 1)

    assert coeffs.shape == (2, dimensions, order+1)
    assert orth.shape == (len(dist), order+1)
    assert norms.shape == (len(dist), order+1)
    return coeffs, orth, norms
