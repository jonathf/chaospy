"""Auto correlation function."""
import numpy

from .pearson import Corr


def Acf(poly, dist, N=None, **kws):
    """
    Auto-correlation function.

    Args:
        poly (numpoly.ndpoly):
            Polynomial of interest. Must have ``len(poly) > N``.
        dist (Distribution):
            Defines the space the correlation is taken on.
        N (int):
            The number of time steps apart included. If omitted set to
            ``len(poly)/2+1``.

    Returns:
        (numpy.ndarray) :
            Auto-correlation of ``poly`` with shape ``(N,)``. Note that by
            definition ``Q[0]=1``.

    Examples:
        >>> poly = chaospy.prange(10)[1:]
        >>> dist = chaospy.Uniform()
        >>> chaospy.Acf(poly, dist, 5).round(4)
        array([1.    , 0.9915, 0.9722, 0.9457, 0.9127])

    """
    if N is None:
        N = len(poly)/2 + 1

    corr = Corr(poly, dist, **kws)
    out = numpy.empty(N)

    for n in range(N):
        out[n] = numpy.mean(corr.diagonal(n), 0)

    return out
