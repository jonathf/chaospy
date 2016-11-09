"""
Implementation of probabilistic collocation method.

Here subsamples of the Golub-weightselsch method is removed at random and weights
renormalized.
"""
import numpy
import chaospy.quad


def probabilistic_collocation(order, dist, subset=.1):
    """
    Probabilistic collocation method.

    Args:
        order (int, array_like) : Quadrature order along each axis.
        dist (Dist) : Distribution to generate samples from.
        subset (float) : Rate of which to removed samples.
    """
    abscissas, weights = chaospy.quad.collection.golub_welsch(order, dist)

    likelihood = dist.pdf(abscissas)

    alpha = numpy.random.random(len(weights))
    alpha = likelihood > alpha*subset*numpy.max(likelihood)

    abscissas = abscissas.T[alpha].T
    weights = weights[alpha]
    return abscissas, weights
