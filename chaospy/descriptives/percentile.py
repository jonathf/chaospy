import numpy
import numpoly


def Perc(poly, q, dist, sample=10000, **kws):
    """
    Percentile function.

    Note that this function is an empirical function that operates using Monte
    Carlo sampling.

    Args:
        poly (Poly):
            Polynomial of interest.
        q (numpy.ndarray):
            positions where percentiles are taken. Must be a number or an
            array, where all values are on the interval ``[0, 100]``.
        dist (Dist):
            Defines the space where percentile is taken.
        sample (int):
            Number of samples used in estimation.

    Returns:
        (numpy.ndarray):
            Percentiles of ``poly`` with ``Q.shape=poly.shape+q.shape``.

    Examples:
        >>> dist = chaospy.J(chaospy.Gamma(1, 1), chaospy.Normal(0, 2))
        >>> x, y = numpoly.symbols("x y")
        >>> poly = numpoly.polynomial([0.05*x, 0.2*y, 0.01*x*y])
        >>> print(chaospy.Perc(poly, [0, 5, 50, 95, 100], dist).round(2))
        [[ 0.000e+00 -3.000e+00 -2.646e+02]
         [ 0.000e+00 -6.400e-01 -8.000e-02]
         [ 3.000e-02 -1.000e-02 -0.000e+00]
         [ 1.500e-01  6.600e-01  8.000e-02]
         [ 2.100e+00  3.000e+00  2.646e+02]]
    """
    shape = poly.shape
    poly = poly.flatten()

    q = numpy.array(q)/100.
    dim = len(dist)

    # Interior
    Z = dist.sample(sample, **kws)
    if dim == 1:
        Z = (Z,)
        q = numpy.array([q])
    poly1 = poly(*Z)

    # Min/max
    mi, ma = dist.range().reshape(2, dim)
    ext = numpy.mgrid[(slice(0, 2, 1), )*dim].reshape(dim, 2**dim).T
    ext = numpy.where(ext, mi, ma).T
    poly2 = poly(*ext)
    poly2 = numpy.array([_ for _ in poly2.T if not numpy.any(numpy.isnan(_))]).T

    # Finish
    if poly2.shape:
        poly1 = numpy.concatenate([poly1, poly2], -1)
    samples = poly1.shape[-1]
    poly1.sort()
    out = poly1.T[numpy.asarray(q*(samples-1), dtype=int)]
    out = out.reshape(q.shape + shape)

    return out
