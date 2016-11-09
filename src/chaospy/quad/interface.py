import inspect

import numpy as np
from scipy.misc import comb

import chaospy
import chaospy.quad


def generate_quadrature(order, domain, accuracy=100, sparse=False, rule="C",
                        composite=1, growth=None, part=None, **kws):
    """
    Numerical quadrature node and weight generator.

    Args:
        order (int) : The order of the quadrature.
        domain (array_like, Dist) : If array is provided domain is the lower
        and upper bounds (lo,up). Invalid if gaussian is set.  If Dist is
            provided, bounds and nodes are adapted to the distribution. This
            includes weighting the nodes in Clenshaw-Curtis quadrature.
        accuracy (int) : If gaussian is set, but the Dist provieded in domain
            does not provide an analytical TTR, ac sets the approximation order
            for the descitized Stieltje's method.
        sparse (bool) : If True used Smolyak's sparse grid instead of normal
            tensor product grid.
        rule (str) : Rule for generating abscissas and weights. Either done
            with quadrature rules, or with random samples with constant
            weights.  For a description of the rules, see
            :py:`chaospy.quad.collection`.
        composite (int, optional) : If provided, composite quadrature will be
            used.  Value determines the number of domains along an axis.
            Ignored in the case gaussian=True.
        growth (bool, optional) : If True sets the growth rule for the
            composite quadrature rule to exponential for Clenshaw-Curtis
            quadrature.
        **kws (optional) : Extra keywords passed to samplegen.
    """
    if growth and order:
        if isinstance(order, int):
            order = 2**order
        else:
            order = tuple([2**o for o in order])

    isdist = isinstance(domain, chaospy.dist.Dist)
    if isdist:
        dim = len(domain)
    else:
        dim = np.array(domain[0]).size

    rule = rule.lower()
    if len(rule) == 1:
        rule = chaospy.quad.collection.QUAD_SHORT_NAMES[rule]

    quad_function = chaospy.quad.collection.get_function(
        rule,
        domain,
        growth=growth,
        composite=composite,
        accuracy=accuracy,
    )

    if sparse:
        order = np.ones(len(domain), dtype=int)*order
        abscissas, weights = chaospy.quad.sparse_grid(
            quad_function, order, dim)

    else:
        abscissas, weights = quad_function(order)

    assert len(weights) == abscissas.shape[1]
    assert len(abscissas.shape) == 2

    if isdist and not sparse:
        abscissas, weights = normalize_weights(abscissas, weights)

    return abscissas, weights


def normalize_weights(abscissas, weights):
    """
    Clean up abscissas and weights.

    Ensure weight sum is 1. Remove entries for infintesimal small weigths.
    """
    weights_sum = np.sum(weights)

    eps = 1
    while (weights_sum - np.sum(weights[weights > eps])) > 1e-18:
        eps *= .1

    valid = weights > eps
    abscissas, weights = abscissas[:, valid], weights[valid]
    weights /= np.sum(weights)
    return abscissas, weights
