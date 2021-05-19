"""Smolyak sparse grid constructor."""
from collections import defaultdict
from itertools import product

import numpy
from scipy.special import comb

import numpoly
import chaospy


def sparse_grid(
        order,
        dist,
        growth=None,
        recurrence_algorithm="stieltjes",
        rule="gaussian",
        tolerance=1e-10,
        scaling=3,
        n_max=5000,
):
    """
    Smolyak sparse grid constructor.

    Args:
        order (int, numpy.ndarray):
            The order of the grid. If ``numpy.ndarray``, it overrides both
            ``dim`` and ``skew``.
        dist (chaospy.distributions.baseclass.Distribution):
            The distribution which density will be used as weight function.
        growth (bool, None):
            If True sets the growth rule for the quadrature rule to only
            include orders that enhances nested samples. Defaults to the same
            value as ``sparse`` if omitted.
        recurrence_algorithm (str):
            Name of the algorithm used to generate abscissas and weights in
            case of Gaussian quadrature scheme. If omitted, ``analytical`` will
            be tried first, and ``stieltjes`` used if that fails.
        rule (str):
            Rule for generating abscissas and weights. Either done with
            quadrature rules, or with random samples with constant weights.
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
        (numpy.ndarray, numpy.ndarray):
            Abscissas and weights created from sparse grid rule. Flatten such
            that ``abscissas.shape == (len(dist), len(weights))``.

    Example:
        >>> distribution = chaospy.J(chaospy.Normal(0, 1), chaospy.Uniform(-1, 1))
        >>> abscissas, weights = chaospy.quadrature.sparse_grid(1, distribution)
        >>> abscissas.round(4)
        array([[-1.    ,  0.    ,  0.    ,  0.    ,  1.    ],
               [ 0.    , -0.5774,  0.    ,  0.5774,  0.    ]])
        >>> weights.round(4)
        array([ 0.5,  0.5, -1. ,  0.5,  0.5])
        >>> abscissas, weights = chaospy.quadrature.sparse_grid([2, 1], distribution)
        >>> abscissas.round(2)
        array([[-1.73, -1.  , -1.  , -1.  ,  0.  ,  1.  ,  1.  ,  1.  ,  1.73],
               [ 0.  , -0.58,  0.  ,  0.58,  0.  , -0.58,  0.  ,  0.58,  0.  ]])
        >>> weights.round(2)
        array([ 0.17,  0.25, -0.5 ,  0.25,  0.67,  0.25, -0.5 ,  0.25,  0.17])
    """
    orders = order*numpy.ones(len(dist), dtype=int)
    growth = True if growth is None else growth

    assert isinstance(dist, chaospy.Distribution), "dist must be chaospy.Distribution"
    dist = dist if isinstance(dist, (chaospy.J, chaospy.Iid)) else chaospy.J(dist)

    if isinstance(rule, str):
        rule = (rule,)*len(dist)

    x_lookup, w_lookup = _construct_lookup(
        orders=orders,
        dists=dist,
        growth=growth,
        recurrence_algorithm=recurrence_algorithm,
        rules=rule,
        tolerance=tolerance,
        scaling=scaling,
        n_max=n_max,
    )
    collection = _construct_collection(
        order, dist, x_lookup, w_lookup)

    abscissas = sorted(collection)
    weights = numpy.array([collection[key] for key in abscissas])
    abscissas = numpy.array(abscissas).T
    return abscissas, weights


def _construct_collection(
        orders,
        dist,
        x_lookup,
        w_lookup,
):
    """Create a collection of {abscissa: weight} key-value pairs."""
    order = numpy.min(orders)
    skew = orders-order

    # Indices and coefficients used in the calculations
    indices = numpoly.glexindex(
        order-len(dist)+1, order+1, dimensions=len(dist))
    coeffs = numpy.sum(indices, -1)
    coeffs = (2*((order-coeffs+1) % 2)-1)*comb(len(dist)-1, order-coeffs)

    collection = defaultdict(float)
    for bidx, coeff in zip(indices+skew, coeffs.tolist()):
        abscissas = [value[idx] for idx, value in zip(bidx, x_lookup)]
        weights = [value[idx] for idx, value in zip(bidx, w_lookup)]
        for abscissa, weight in zip(product(*abscissas), product(*weights)):
            collection[abscissa] += numpy.prod(weight)*coeff

    return collection


def _construct_lookup(
        orders,
        dists,
        growth,
        recurrence_algorithm,
        rules,
        tolerance,
        scaling,
        n_max,
):
    """
    Create abscissas and weights look-up table so values do not need to be
    re-calculatated on the fly.
    """
    x_lookup = []
    w_lookup = []
    for max_order, dist, rule in zip(orders, dists, rules):
        x_lookup.append([])
        w_lookup.append([])
        for order in range(max_order+1):
            (abscissas,), weights = chaospy.generate_quadrature(
                order=order,
                dist=dist,
                growth=growth,
                recurrence_algorithm=recurrence_algorithm,
                rule=rule,
                tolerance=tolerance,
                scaling=scaling,
                n_max=n_max,
            )
            x_lookup[-1].append(abscissas)
            w_lookup[-1].append(weights)
    return x_lookup, w_lookup
