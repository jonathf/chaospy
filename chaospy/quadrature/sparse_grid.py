"""
As the number of dimensions increases linear, the number of samples increases
exponentially. This is known as the curse of dimensionality. Except for
switching to Monte Carlo integration, the is no way to completly guard against
this problem. However, there are some possibility to mitigate the problem
personally. One such strategy is to employ Smolyak sparse-grid quadrature. This
method uses a quadrature rule over a combination of different orders to tailor
a scheme that uses fewer abscissas points than a full tensor-product approach.

To use Smolyak sparse-grid in ``chaospy``, just pass the flag ``sparse=True``
to the ``generate_quadrature`` function. For example::

    >>> distribution = chaospy.J(
    ...     chaospy.Uniform(0, 4), chaospy.Uniform(0, 4))
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     3, distribution, sparse=True)
    >>> abscissas.round(4)
    array([[0., 0., 0., 1., 2., 2., 2., 2., 2., 3., 4., 4., 4.],
           [0., 2., 4., 2., 0., 1., 2., 3., 4., 2., 0., 2., 4.]])
    >>> weights.round(4)
    array([-0.0833,  0.2222, -0.0833,  0.4444,  0.2222,  0.4444, -1.3333,
            0.4444,  0.2222,  0.4444, -0.0833,  0.2222, -0.0833])

This compared to the full tensor-product grid::

    >>> abscissas, weights = chaospy.generate_quadrature(3, distribution, sparse=False)
    >>> abscissas.round(4)
    array([[0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 3., 3., 4., 4., 4., 4.],
           [0., 1., 3., 4., 0., 1., 3., 4., 0., 1., 3., 4., 0., 1., 3., 4.]])
    >>> weights.round(4)
    array([0.0031, 0.0247, 0.0247, 0.0031, 0.0247, 0.1975, 0.1975, 0.0247,
           0.0247, 0.1975, 0.1975, 0.0247, 0.0031, 0.0247, 0.0247, 0.0031])

The method works with all quadrature rules, but is known to be quite
inefficient when applied to rules that can not be nested. For example using
Gauss-Legendre samples::

    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     6, distribution, rule="gauss_legendre", sparse=True)
    >>> len(weights)
    140
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     6, distribution, rule="gauss_legendre", sparse=False)
    >>> len(weights)
    49

.. note:
    Some quadrature rules are only partially nested at certain orders. These
    include e.g. :ref:`clenshaw_curtis`, :ref:`fejer` and :ref:`newton_cotes`.
    To exploit this nested-nes, the default behavior is to only include orders
    that are properly nested. This implies that flipping the flag ``sparse``
    will result in a somewhat different scheme. To fix the scheme one way or
    the other, explicitly include the flag ``growth=False`` or ``growth=True``
    respectively.
"""
from collections import defaultdict
from itertools import product

import numpy
from scipy.special import comb
import numpoly
import chaospy


def construct_sparse_grid(
        order,
        dist,
        rule="gaussian",
        accuracy=100,
        growth=None,
        recurrence_algorithm="",
):
    """
    Smolyak sparse grid constructor.

    Args:
        order (int, numpy.ndarray):
            The order of the grid. If ``numpy.ndarray``, it overrides both
            ``dim`` and ``skew``.
        dist (chaospy.distributions.baseclass.Distribution):
            The distribution which density will be used as weight function.
        rule (str):
            Rule for generating abscissas and weights. Either done with
            quadrature rules, or with random samples with constant weights.
        accuracy (int):
            If gaussian is set, but the dist provided in domain does not
            provide an analytical TTR, ac sets the approximation order for the
            descitized Stieltje's method.
        growth (bool, None):
            If True sets the growth rule for the quadrature rule to only
            include orders that enhances nested samples. Defaults to the same
            value as ``sparse`` if omitted.
        recurrence_algorithm (str):
            Name of the algorithm used to generate abscissas and weights in
            case of Gaussian quadrature scheme. If omitted, ``analytical`` will
            be tried first, and ``stieltjes`` used if that fails.

    Returns:
        (numpy.ndarray, numpy.ndarray):
            Abscissas and weights created from sparse grid rule. Flatten such
            that ``abscissas.shape == (len(dist), len(weights))``.

    Example:
        >>> distribution = chaospy.J(
        ...     chaospy.Normal(0, 1), chaospy.Uniform(-1, 1))
        >>> abscissas, weights = construct_sparse_grid(1, distribution)
        >>> abscissas.round(4)
        array([[-1.    ,  0.    ,  0.    ,  0.    ,  1.    ],
               [ 0.    , -0.5774,  0.    ,  0.5774,  0.    ]])
        >>> weights.round(4)
        array([ 0.5,  0.5, -1. ,  0.5,  0.5])
        >>> abscissas, weights = construct_sparse_grid([2, 1], distribution)
        >>> abscissas.round(2)
        array([[-1.73, -1.  , -1.  , -1.  ,  0.  ,  1.  ,  1.  ,  1.  ,  1.73],
               [ 0.  , -0.58,  0.  ,  0.58,  0.  , -0.58,  0.  ,  0.58,  0.  ]])
        >>> weights.round(2)
        array([ 0.17,  0.25, -0.5 ,  0.25,  0.67,  0.25, -0.5 ,  0.25,  0.17])
    """
    orders = order*numpy.ones(len(dist), dtype=int)

    assert isinstance(dist, chaospy.Distribution), "dist must be chaospy.Distribution"
    if not isinstance(dist, (chaospy.J, chaospy.Iid)):
        dist = chaospy.J(dist)

    if isinstance(rule, str):
        rule = (rule,)*len(dist)

    x_lookup, w_lookup = _construct_lookup(
        orders, dist, rule, accuracy, growth, recurrence_algorithm)
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
        rules,
        accuracy,
        growth,
        recurrence_algorithm,
):
    """
    Create abscissas and weights look-up table so values do not need to be
    re-calculatated on the fly.
    """
    from .frontend import generate_quadrature
    x_lookup = []
    w_lookup = []
    for max_order, dist, rule in zip(orders, dists, rules):
        x_lookup.append([])
        w_lookup.append([])
        for order in range(max_order+1):
            (abscissas,), weights = generate_quadrature(
                order,
                dist,
                accuracy=accuracy,
                rule=rule,
                growth=growth,
                recurrence_algorithm=recurrence_algorithm,
            )
            x_lookup[-1].append(abscissas)
            w_lookup[-1].append(weights)
    return x_lookup, w_lookup
