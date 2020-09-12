"""Spearman's correlation coefficient."""
from scipy.stats import spearmanr


def Spearman(poly, dist, sample=10000, retall=False, **kws):
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
    """
    samples = dist.sample(sample, **kws)
    poly = polynomials.flatten(poly)
    Y = poly(*samples)
    if retall:
        return spearmanr(Y.T)
    return spearmanr(Y.T)[0]
