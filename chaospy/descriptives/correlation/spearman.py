"""Spearman's correlation coefficient."""
import numpy
from scipy.stats import spearmanr

import numpoly
import chaospy


def Spearman(poly, dist=None, sample=10000, retall=False, **kws):
    """
    Calculate Spearman's rank-order correlation coefficient.

    Args:
        poly (numpoly.ndpoly):
            Polynomial of interest.
        dist (Distribution):
            Defines the space where correlation is taken.
        sample (int):
            Number of samples used in estimation.
        retall (bool):
            If true, return p-value as well.

    Returns:
        (float, numpy.ndarray):
            Correlation output ``rho``. Of type float if two-dimensional problem.
            Correleation matrix if larger.
        (float, numpy.ndarray):
            The two-sided p-value for a hypothesis test whose null hypothesis
            is that two sets of data are uncorrelated, has same dimension as
            ``rho``.

    Examples:
        >>> distribution = chaospy.MvNormal(
        ...     [3, 4], [[2, .5], [.5, 1]])
        >>> corr, pvalue = chaospy.Spearman(distribution, sample=50, retall=True)
        >>> corr.round(4)
        array([[1.   , 0.603],
               [0.603, 1.   ]])
        >>> pvalue.round(8)
        array([[0.00e+00, 3.58e-06],
               [3.58e-06, 0.00e+00]])

    """
    if isinstance(poly, chaospy.Distribution):
        poly, dist = numpoly.variable(len(poly)), poly
    else:
        poly = numpoly.polynomial(poly)
    samples = dist.sample(sample, **kws)
    corr = numpy.eye(len(poly))
    pval = numpy.zeros((len(poly), len(poly)))
    evals = poly.ravel()(*samples)
    assert len(poly) == len(evals)
    for idx in range(len(poly)):
        for idy in range(idx+1, len(poly)):
            if idx == idy:
                pass
            spear = spearmanr(evals[idx], evals[idy])
            pval[idx, idy] = pval[idy, idx] = spear.pvalue
            corr[idx, idy] = corr[idy, idx] = spear.correlation
    if retall:
        return corr, pval
    return corr
