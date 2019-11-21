"""Auto correlation function."""
import numpy

from .pearson import Corr


def Acf(poly, dist, N=None, **kws):
    """
    Auto-correlation function.

    Args:
        poly (chaospy.poly.ndpoly):
            Polynomial of interest. Must have ``len(poly) > N``.
        dist (Dist):
            Defines the space the correlation is taken on.
        N (int):
            The number of time steps appart included. If omited set to
            ``len(poly)/2+1``.

    Returns:
        (numpy.ndarray) :
            Auto-correlation of ``poly`` with shape ``(N,)``. Note that by
            definition ``Q[0]=1``.

    Examples:
        >>> poly = chaospy.prange(10)[1:]
        >>> Z = chaospy.Uniform()
        >>> print(numpy.around(chaospy.Acf(poly, Z, 5), 4))
        [1.     0.9915 0.9722 0.9457 0.9127]
    """
    if N is None:
        N = len(poly)/2 + 1

    corr = Corr(poly, dist, **kws)
    out = numpy.empty(N)

    for n in range(N):
        out[n] = numpy.mean(corr.diagonal(n), 0)

    return out
