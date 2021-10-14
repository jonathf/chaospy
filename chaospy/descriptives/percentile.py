"""Percentile function."""
import numpy
import chaospy


def Perc(poly, q, dist, sample=10000, **kws):
    """
    Percentile function.

    Note that this function is an empirical function that operates using Monte
    Carlo sampling.

    Args:
        poly (numpoly.ndpoly):
            Polynomial of interest.
        q (numpy.ndarray):
            positions where percentiles are taken. Must be a number or an
            array, where all values are on the interval ``[0, 100]``.
        dist (Distribution):
            Defines the space where percentile is taken.
        sample (int):
            Number of samples used in estimation.

    Returns:
        (numpy.ndarray):
            Percentiles of ``poly`` with ``Q.shape=poly.shape+q.shape``.

    Examples:
        >>> dist = chaospy.J(chaospy.Gamma(1, 1), chaospy.Normal(0, 2))
        >>> q0, q1 = chaospy.variable(2)
        >>> poly = chaospy.polynomial([0.05*q0, 0.2*q1, 0.01*q0*q1])
        >>> chaospy.Perc(poly, [0, 5, 50, 95, 100], dist).round(2)
        array([[ 0.  , -3.29, -5.3 ],
               [ 0.  , -0.64, -0.04],
               [ 0.03, -0.01, -0.  ],
               [ 0.15,  0.66,  0.04],
               [ 1.61,  3.29,  5.3 ]])

    """
    poly = chaospy.aspolynomial(poly)
    shape = poly.shape
    poly = poly.ravel()

    q = numpy.asarray(q).ravel()/100.
    dim = len(dist)

    # Interior
    Z = dist.sample(sample, **kws).reshape(len(dist), sample)
    poly1 = poly(*Z)

    # Min/max
    ext = numpy.mgrid[(slice(0, 2, 1), )*dim].reshape(dim, 2**dim).T
    ext = numpy.where(ext, dist.lower, dist.upper).T
    poly2 = poly(*ext)
    poly2 = numpy.array([_ for _ in poly2.T if not numpy.any(numpy.isnan(_))]).T

    # Finish
    if numpy.sum(poly2.shape) > 0:
        poly1 = numpy.concatenate([poly1, poly2], -1)
    samples = poly1.shape[-1]
    poly1.sort()
    out = poly1.T[numpy.asarray(q*(samples-1), dtype=int)]
    out = out.reshape(q.shape + shape)

    return out
