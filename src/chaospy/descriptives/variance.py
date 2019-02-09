"""Variance operator."""
import numpy

from .. import distributions, poly as polynomials


def Var(poly, dist=None, **kws):
    """
    Element by element 2nd order statistics.

    Args:
        poly (Poly, Dist):
            Input to take variance on.
        dist (Dist):
            Defines the space the variance is taken on. It is ignored if
            ``poly`` is a distribution.

    Returns:
        (numpy.ndarray):
            Element for element variance along ``poly``, where
            ``variation.shape == poly.shape``.

    Examples:
        >>> dist = chaospy.J(chaospy.Gamma(1, 1), chaospy.Normal(0, 2))
        >>> print(chaospy.Var(dist))
        [1. 4.]
        >>> x, y = chaospy.variable(2)
        >>> poly = chaospy.Poly([1, x, y, 10*x*y])
        >>> print(chaospy.Var(poly, dist))
        [  0.   1.   4. 800.]
    """
    if isinstance(poly, distributions.Dist):
        x = polynomials.variable(len(poly))
        poly, dist = x, poly
    else:
        poly = polynomials.Poly(poly)

    dim = len(dist)
    if poly.dim<dim:
        polynomials.setdim(poly, dim)

    shape = poly.shape
    poly = polynomials.flatten(poly)

    keys = poly.keys
    N = len(keys)
    A = poly.A

    keys1 = numpy.array(keys).T
    if dim==1:
        keys1 = keys1[0]
        keys2 = sum(numpy.meshgrid(keys, keys))
    else:
        keys2 = numpy.empty((dim, N, N))
        for i in range(N):
            for j in range(N):
                keys2[:, i, j] = keys1[:, i]+keys1[:, j]

    m1 = numpy.outer(*[dist.mom(keys1, **kws)]*2)
    m2 = dist.mom(keys2, **kws)
    mom = m2-m1

    out = numpy.zeros(poly.shape)
    for i in range(N):
        a = A[keys[i]]
        out += a*a*mom[i, i]
        for j in range(i+1, N):
            b = A[keys[j]]
            out += 2*a*b*mom[i, j]

    out = out.reshape(shape)
    return out
