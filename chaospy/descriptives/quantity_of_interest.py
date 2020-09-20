"""Constructs distributions for the quantity of interests."""
from __future__ import division
from functools import reduce
from operator import mul
import numpy

from .. import distributions
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
        poly (numpoly.ndpoly):
            Polynomial of interest.
        dist (Distribution):
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

    # reshape the qoi_dists to match the shape of the input poly
    if shape:
        def reshape(lst, shape):
            if len(shape) == 1:
                return lst
            n = reduce(mul, shape[1:])
            return [reshape(lst[i*n:(i+1)*n], shape[1:]) for i in range(len(lst)//n)]
        qoi_dists = reshape(qoi_dists, shape)
    else:
        qoi_dists = qoi_dists[0]

    if not shape:
        qoi_dists = qoi_dists.item()

    return qoi_dists
