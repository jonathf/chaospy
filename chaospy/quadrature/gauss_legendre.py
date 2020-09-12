r"""
The Gauss-Legendre quadrature rule is properly supported by in :ref:`gaussian`.
However, as Gauss-Legendre is a special case where the weight function is
constant, it can in principle be used to integrate any weighting function. In
other words, this is the same Gauss-Legendre integration rule, but only in the
context of uniform distribution as weight function. Normalization of the
weights will be used to achieve the general integration form.

It is also worth noting that this specific implementation of Gauss-Legendre is
faster to compute than the general version in :ref:`gaussian`.

Example usage
-------------

The first few orders::

    >>> distribution = chaospy.Uniform(0, 1)
    >>> for order in [0, 1, 2, 3]:
    ...     abscissas, weights = chaospy.generate_quadrature(
    ...         order, distribution, rule="gauss_legendre")
    ...     print(order, abscissas.round(3), weights.round(3))
    0 [[0.5]] [1.]
    1 [[0.211 0.789]] [0.5 0.5]
    2 [[0.113 0.5   0.887]] [0.278 0.444 0.278]
    3 [[0.069 0.33  0.67  0.931]] [0.174 0.326 0.326 0.174]

Using an alternative distribution::

    >>> distribution = chaospy.Beta(2, 4)
    >>> for order in [0, 1, 2, 3]:
    ...     abscissas, weights = chaospy.generate_quadrature(
    ...         order, distribution, rule="gauss_legendre")
    ...     print(order, abscissas.round(3), weights.round(3))
    0 [[0.5]] [1.]
    1 [[0.211 0.789]] [0.933 0.067]
    2 [[0.113 0.5   0.887]] [0.437 0.556 0.007]
    3 [[0.069 0.33  0.67  0.931]] [0.195 0.647 0.157 0.001]

The abscissas stays the same, but the weights are re-adjusted for the new
weight function.
"""
import numpy

from .recurrence import (
    construct_recurrence_coefficients, coefficients_to_quadrature)
from .combine import combine_quadrature


def quad_gauss_legendre(
        order,
        domain=(0, 1),
        rule="fejer",
        accuracy=100,
        recurrence_algorithm="",
):
    r"""
    Generate the quadrature nodes and weights in Gauss-Legendre quadrature.

    Note that this rule exists to allow for integrating functions with weight
    functions without actually adding the quadrature. Like:

    .. math:
        \int_a^b p(x) f(x) dx \approx \sum_i p(X_i) f(X_i) W_i

    instead of the more traditional:

    .. math:
        \int_a^b p(x) f(x) dx \approx \sum_i f(X_i) W_i

    To get the behavior where the weight function is taken into consideration,
    use :func:`~chaospy.quadrature.gaussian.quad_gaussian`.

    Args:
        order (int, numpy.ndarray):
            Quadrature order.
        domain (chaospy.distributions.baseclass.Distribution, numpy.ndarray):
            Either distribution or bounding of interval to integrate over.
        rule (str):
            In the case of ``lanczos`` or ``stieltjes``, defines the
            proxy-integration scheme.
        accuracy (int):
            In the case ``rule`` is used, defines the quadrature order of the
            scheme used. In practice, must be at least as large as ``order``.
        recurrence_algorithm (str):
            Name of the algorithm used to generate abscissas and weights. If
            omitted, ``analytical`` will be tried first, and ``stieltjes`` used
            if that fails.

    Returns:
        (numpy.ndarray, numpy.ndarray):
            abscissas:
                The quadrature points for where to evaluate the model function
                with ``abscissas.shape == (len(dist), N)`` where ``N`` is the
                number of samples.
            weights:
                The quadrature weights with ``weights.shape == (N,)``.

    Example:
        >>> abscissas, weights = quad_gauss_legendre(3)
        >>> abscissas.round(4)
        array([[0.0694, 0.33  , 0.67  , 0.9306]])
        >>> weights.round(4)
        array([0.1739, 0.3261, 0.3261, 0.1739])
    """
    from ..distributions.baseclass import Distribution
    from ..distributions.collection import Uniform
    if isinstance(domain, Distribution):
        abscissas, weights = quad_gauss_legendre(
            order, (domain.lower, domain.upper),
            rule, accuracy, recurrence_algorithm)

        pdf = domain.pdf(abscissas)
        if len(domain) > 1:
            weights = (weights.T*pdf).T
        else:
            weights *= pdf.flatten()
        weights /= numpy.sum(weights)
        return abscissas, weights

    order = numpy.asarray(order, dtype=int).flatten()
    lower, upper = numpy.array(domain)
    lower = numpy.asarray(lower).flatten()
    upper = numpy.asarray(upper).flatten()

    dim = max(lower.size, upper.size, order.size)
    order = numpy.ones(dim, dtype=int)*order
    lower = numpy.ones(dim)*lower
    upper = numpy.ones(dim)*upper

    coefficients = construct_recurrence_coefficients(
        numpy.max(order), Uniform(0, 1), rule, accuracy, recurrence_algorithm)

    abscissas, weights = zip(*[coefficients_to_quadrature(
        coefficients[:order_+1]) for order_ in order])
    abscissas = list(numpy.asarray(abscissas).reshape(dim, -1))
    weights = list(numpy.asarray(weights).reshape(dim, -1))

    return combine_quadrature(abscissas, weights, (lower, upper))
