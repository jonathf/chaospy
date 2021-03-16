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
    1 [[0.11 0.4 ]] [0. 1.]
    2 [[0.11  0.4   0.787]] [0.203 0.644 0.153]
    3 [[0.11  0.4   0.6   0.787]] [0.22  0.574 0.086 0.12 ]
    4 [[0.027 0.11  0.4   0.6   0.787]] [-0.047  0.298  0.499  0.146  0.103]

"""
import numpy
from scipy.optimize import fminbound
import chaospy

from .utils import combine


def leja(
        order,
        dist,
        rule="fejer_2",
):
    """
    Generate Leja quadrature node.

    Args:
        order (int):
            The order of the quadrature.
        dist (chaospy.distributions.baseclass.Distribution):
            The distribution which density will be used as weight function.
        rule (str):
            In the case of ``lanczos`` or ``stieltjes``, defines the
            proxy-integration scheme.

    Returns:
        (numpy.ndarray, numpy.ndarray):
            abscissas:
                The quadrature points for where to evaluate the model function
                with ``abscissas.shape == (len(dist), N)`` where ``N`` is the
                number of samples.
            weights:
                The quadrature weights with ``weights.shape == (N,)``.

    Notes:
        Implemented as proposed in Narayan and Jakeman
        :cite:`narayan_adaptive_2014`.

    Example:
        >>> distribution = chaospy.Iid(chaospy.Normal(0, 1), 2)
        >>> abscissas, weights = chaospy.quadrature.leja(2, distribution)
        >>> abscissas.round(2)
        array([[-1.41, -1.41, -1.41,  0.  ,  0.  ,  0.  ,  1.76,  1.76,  1.76],
               [-1.41,  0.  ,  1.76, -1.41,  0.  ,  1.76, -1.41,  0.  ,  1.76]])
        >>> weights.round(3)
        array([0.05 , 0.133, 0.04 , 0.133, 0.359, 0.107, 0.04 , 0.107, 0.032])

    """
    if len(dist) > 1:
        if dist.stochastic_dependent:
            raise chaospy.StochasticallyDependentError(
                "Leja quadrature do not supper distribution with dependencies.")
        order = numpy.broadcast_to(order, len(dist))
        out = [leja(order[_], dist[_]) for _ in range(len(dist))]
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

        def fmin(idx):
            """Bound minimization."""
            try:
                xopt, fval, _, _ = fminbound(objective, abscissas[idx],
                                             abscissas[idx+1], full_output=True)
            # Hard coded solution to scipy/scipy#11207 for scipy < 1.5.0.
            except UnboundLocalError:  # pragma: no cover
                xopt = abscissas[idx]+0.5*(3-5**0.5)*(
                    abscissas[idx+1]-abscissas[idx])
                fx = objective(xopt)
            return xopt, fval

        opts, vals = zip(*[fmin(idx) for idx in range(len(abscissas)-1)])
        index = numpy.argmin(vals)
        abscissas.insert(index+1, opts[index])

    abscissas = numpy.asfarray(abscissas).flatten()[1:-1]
    weights = create_weights(abscissas, dist, rule)
    abscissas = abscissas.reshape(1, abscissas.size)

    return numpy.asfarray(abscissas), numpy.asfarray(weights).flatten()


def create_weights(
        nodes,
        dist,
        rule="clenshaw_curtis",
):
    """Create weights for the Laja method."""
    _, poly, _ = chaospy.stieltjes(len(nodes)-1, dist, rule=rule)
    poly = poly.ravel()
    weights = numpy.linalg.inv(poly(nodes))
    return weights[:, 0]
