"""Auto correlation function."""
import numpy

from .pearson import Corr


def Acf(poly, dist, n_steps=None, **kws):
    """
    Auto-correlation function.

    Args:
        poly (numpoly.ndpoly):
            Polynomial of interest. Must have ``len(poly) > n_steps``.
        dist (Distribution):
            Defines the space the correlation is taken on.
        n_steps (int):
            The number of time steps apart included. If omitted set to
            ``len(poly)/2+1``.

    Returns:
        (numpy.ndarray) :
            Auto-correlation of ``poly`` with shape ``(n_steps,)``. Note that
            by definition ``Q[0]=1``.

    Examples:
        >>> poly = chaospy.monomial(1, 10)
        >>> dist = chaospy.Uniform()
        >>> chaospy.Acf(poly, dist).round(4)
        array([1.    , 0.9915, 0.9722, 0.9457, 0.9127])

    """
    n_steps = int(len(poly)/2+1) if n_steps is None else n_steps
    correlation = Corr(poly, dist, **kws)
    return numpy.array([numpy.mean(correlation.diagonal(idx), axis=0)
                        for idx in range(n_steps)])
