"""Frontend for the generation of quadrature rules."""
import inspect

import numpy as np
from scipy.special import comb

from . import collection, sparse_grid


def generate_quadrature(
        order, domain, accuracy=100, sparse=False, rule="C",
        growth=None, normalize=False, **kws
):
    """
    Numerical quadrature node and weight generator.

    Args:
        order (int):
            The order of the quadrature.
        domain (numpy.ndarray, Dist):
            If array is provided domain is the lower and upper bounds (lo,up).
            Invalid if gaussian is set.  If Dist is provided, bounds and nodes
            are adapted to the distribution. This includes weighting the nodes
            in Clenshaw-Curtis quadrature.
        accuracy (int):
            If gaussian is set, but the Dist provieded in domain does not
            provide an analytical TTR, ac sets the approximation order for the
            descitized Stieltje's method.
        sparse (bool):
            If True used Smolyak's sparse grid instead of normal tensor product
            grid.
        rule (str):
            Rule for generating abscissas and weights. Either done with
            quadrature rules, or with random samples with constant weights.
        normalize (bool):
            In the case of distributions, the abscissas and weights are not
            tailored to a distribution beyond matching the bounds. If True, the
            samples are normalized multiplying the weights with the density of
            the distribution evaluated at the abscissas and normalized
            afterwards to sum to one.
        growth (bool, None):
            If True sets the growth rule for the quadrature rule to only
            include orders that enhances nested samples. Defaults to the same
            value as ``sparse`` if omitted.
    """
    from ..distributions.baseclass import Dist
    isdist = isinstance(domain, Dist)
    if isdist:
        dim = len(domain)
    else:
        dim = np.array(domain[0]).size

    rule = rule.lower()
    if len(rule) == 1:
        rule = collection.QUAD_SHORT_NAMES[rule]

    quad_function = collection.get_function(
        rule,
        domain,
        normalize,
        growth=growth,
        accuracy=accuracy,
    )

    if sparse:
        order = np.ones(len(domain), dtype=int)*order
        abscissas, weights = sparse_grid.sparse_grid(quad_function, order, dim)

    else:
        abscissas, weights = quad_function(order)

    assert len(weights) == abscissas.shape[1]
    assert len(abscissas.shape) == 2

    return abscissas, weights
