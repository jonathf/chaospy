import numpy

from .. import distributions, poly as polynomials
from ..external import SampleDist


def QoI_Dist(poly, dist, sample=10000, **kws):
    """
    Constructs distributions for the quantity of interests.

    The function constructs a kernel density estimator (KDE) for each
    polynomial (poly) by sampling it.  With the KDEs, distributions (Dists) are
    constructed.  The Dists can be used for e.g. plotting probability density
    functions (PDF), or to make a second uncertainty quantification simulation
    with that newly generated Dists.

    Args:
        poly (chaospy.poly.ndpoly):
            Polynomial of interest.
        dist (Dist):
            Defines the space where the samples for the KDE is taken from the
            poly.
        sample (int):
            Number of samples used in estimation to construct the KDE.

    Returns:
        (numpy.ndarray):
            The constructed quantity of interest (QoI) distributions, where
            ``qoi_dists.shape==poly.shape``.

    Examples:
        >>> dist = chaospy.Normal(0, 1)
        >>> x = chaospy.variable(1)
        >>> poly = chaospy.polynomial([x])
        >>> qoi_dist = chaospy.QoI_Dist(poly, dist)
        >>> values = qoi_dist[0].pdf([-0.75, 0., 0.75])
        >>> values.round(8)
        array([0.29143037, 0.39931708, 0.29536329])
    """
    shape = poly.shape
    poly = poly.flatten()
    dim = len(dist)

    #sample from the inumpyut dist
    samples = dist.sample(sample, **kws)

    qoi_dists = []
    for i in range(0, len(poly)):
        #sample the polynomial solution
        if dim == 1:
            dataset = poly[i](samples)
        else:
            dataset = poly[i](*samples)

        lo = dataset.min()
        up = dataset.max()

        #creates qoi_dist
        qoi_dist = SampleDist(dataset, lo, up)
        qoi_dists.append(qoi_dist)

    #reshape the qoi_dists to match the shape of the inumpyut poly
    qoi_dists = numpy.array(qoi_dists, distributions.Dist)
    qoi_dists = qoi_dists.reshape(shape)

    if not shape:
        qoi_dists = qoi_dists.item()

    return qoi_dists
