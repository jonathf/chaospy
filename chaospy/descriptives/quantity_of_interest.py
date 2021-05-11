"""Constructs distributions for the quantity of interests."""
from __future__ import division
import numpy
import chaospy


def QoI_Dist(poly, dist, sample=10000, **kws):
    """
    Constructs distributions for the quantity of interests.

    The function constructs a kernel density estimator (KDE) for each
    polynomial (poly) by sampling it.  With the KDEs, distributions (Dists) are
    constructed.  The Dists can be used for e.g. plotting probability density
    functions (PDF), or to make a second uncertainty quantification simulation
    with that newly generated Dists.

    Args:
        poly (numpoly.ndpoly):
            Polynomial of interest.
        dist (Distribution):
            Defines the space where the samples for the KDE is taken from the
            poly.
        sample (int):
            Number of samples used in estimation to construct the KDE.

    Returns:
        (Distribution):
            The constructed quantity of interest (QoI) distributions, where
            ``qoi_dists.shape==poly.shape``.

    Examples:
        >>> dist = chaospy.Normal(0, 1)
        >>> x = chaospy.variable(1)
        >>> poly = chaospy.polynomial([x])
        >>> qoi_dist = chaospy.QoI_Dist(poly, dist)
        >>> values = qoi_dist.pdf([-0.75, 0., 0.75])
        >>> values.round(8)
        array([0.29143989, 0.39939823, 0.29531414])

    """
    poly = chaospy.aspolynomial(poly).ravel()
    samples = numpy.atleast_2d(dist.sample(sample, **kws))
    qoi_dist = chaospy.GaussianKDE(poly(*samples))
    return qoi_dist
