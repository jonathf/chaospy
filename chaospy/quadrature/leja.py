"""
Laja quadrature is a newer method for performing quadrature in stochastical
problems. The method is described in a `journal paper`_ by Narayan and Jakeman.

.. _journal paper: https://arxiv.org/pdf/1404.5663.pdf

Example usage
-------------

The first few orders::

    >>> distribution = chaospy.Beta(2, 3)
    >>> for order in [0, 1, 2, 3, 4]:  # doctest: +NORMALIZE_WHITESPACE
    ...     abscissas, weights = chaospy.generate_quadrature(
    ...         order, distribution, rule="leja")
    ...     print(order, abscissas.round(3), weights.round(3))
    0 [[0.4]] [1.]
    1 [[0.11 0.4 ]] [ 0.    -3.443]
    2 [[0.11  0.4   0.787]] [ 0.203 -1.676  5.081]
    3 [[0.11  0.4   0.6   0.787]] [  0.22   -1.431   4.7   -10.352]
    4 [[0.027 0.11  0.4   0.6   0.787]]
       [-4.7000e-02  2.7300e-01  2.0220e+00 -5.8250e+00  7.3986e+01]
"""
import numpy
from scipy.optimize import fminbound

from .combine import combine
from .recurrence import analytical_stieljes, discretized_stieltjes


def quad_leja(
        order,
        dist,
        rule="fejer",
        accuracy=100,
        recurrence_algorithm="",
):
    """
    Generate Leja quadrature node.

    Args:
        order (int):
            The order of the quadrature.
        dist (chaospy.distributions.baseclass.Dist):
            The distribution which density will be used as weight function.
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
        >>> abscissas, weights = quad_leja(3, chaospy.Normal(0, 1))
        >>> abscissas.round(4)
        array([[-2.7173, -1.4142,  0.    ,  1.7635]])
        >>> weights.round(4)
        array([ 0.022 , -0.0319,  0.022 , -0.063 ])
    """
    from chaospy.distributions import evaluation

    if len(dist) > 1:
        if evaluation.get_dependencies(*list(dist)):
            raise evaluation.DependencyError(
                "Leja quadrature do not supper distribution with dependencies.")
        if isinstance(order, int):
            out = [quad_leja(order, _) for _ in dist]
        else:
            out = [quad_leja(order[_], dist[_]) for _ in range(len(dist))]

        abscissas = [_[0][0] for _ in out]
        weights = [_[1] for _ in out]
        abscissas = combine(abscissas).T
        weights = combine(weights)
        weights = numpy.prod(weights, -1)

        return abscissas, weights

    abscissas = [dist.lower, dist.mom(1).flatten(), dist.upper]
    for _ in range(int(order)):

        def objective(abscissas_):
            """Local objective function."""
            out = -numpy.sqrt(dist.pdf(abscissas_))*numpy.prod(
                numpy.abs(abscissas[1:-1]-abscissas_))
            return out

        opts, vals = zip(
            *[fminbound(
                objective, abscissas[idx], abscissas[idx+1], full_output=1)[:2]
              for idx in range(len(abscissas)-1)]
        )
        index = numpy.argmin(vals)
        abscissas.insert(index+1, opts[index])

    abscissas = numpy.asfarray(abscissas).flatten()[1:-1]
    weights = create_weights(
        abscissas, dist, rule, accuracy, recurrence_algorithm)
    abscissas = abscissas.reshape(1, abscissas.size)

    return numpy.asfarray(abscissas), numpy.asfarray(weights).flatten()


def create_weights(
        nodes,
        dist,
        rule="fejer",
        accuracy=100,
        recurrence_algorithm="",
):
    """Create weights for the Laja method."""
    try:
        _, poly, _ = analytical_stieljes(len(nodes)-1, dist)
    except NotImplementedError:
        from .frontend import generate_quadrature
        abscissas, weights = generate_quadrature(
            order=accuracy, dist=dist, rule=rule,
            recurrence_algorithm=recurrence_algorithm,
        )
        _, poly, _ = discretized_stieltjes(len(nodes)-1, abscissas, weights)
    weights = numpy.linalg.inv(poly(nodes))
    return weights[:, 0]
