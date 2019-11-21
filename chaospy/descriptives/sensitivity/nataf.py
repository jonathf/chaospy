"""Variance based decomposition on the Nataf Copula."""
import numpy

import chaospy

from ..conditional import E_cond
from ..variance import Var


def Sens_m_nataf(order, dist, samples, vals, **kws):
    """
    Variance-based decomposition through the Nataf distribution.

    Generates first order sensitivity indices.

    Args:
        order (int):
            Polynomial order used ``orth_ttr``.
        dist (chaospy.distributions.copula.nataf.Nataf):
            Assumed to be Nataf with independent components
        samples (numpy.ndarray):
            Samples used for evaluation (typically generated from ``dist``.)
        vals (numpy.ndarray):
            Evaluations of the model for given samples.

    Returns:
        (numpy.ndarray):
            Sensitivity indices with shape ``(len(dist),) + vals.shape[1:]``.
    """
    assert dist.__class__.__name__ == "Copula"
    trans = dist.prm["trans"]
    assert trans.__class__.__name__ == "nataf"
    vals = numpy.array(vals)

    cov = trans.prm["C"]
    cov = numpy.dot(cov, cov.T)

    marginal = dist.prm["dist"]
    dim = len(dist)

    orth = chaospy.orthogonal.orth_ttr(order, marginal, sort="GR")

    r = range(dim)

    index = [1] + [0]*(dim-1)

    nataf = chaospy.dist.Nataf(marginal, cov, r)
    samples_ = marginal.inv( nataf.fwd( samples ) )
    poly, coeffs = chaospy.collocation.fit_regression(
        orth, samples_, vals, retall=1)

    V = Var(poly, marginal, **kws)

    out = numpy.zeros((dim,) + poly.shape)
    out[0] = Var(E_cond(poly, index, marginal, **kws),
                 marginal, **kws)/(V+(V == 0))*(V != 0)


    for i in range(1, dim):

        r = r[1:] + r[:1]
        index = index[-1:] + index[:-1]

        nataf = chaospy.dist.Nataf(marginal, cov, r)
        samples_ = marginal.inv( nataf.fwd( samples ) )
        poly, coeffs = chaospy.collocation.fit_regression(
            orth, samples_, vals, retall=1)

        out[i] = Var(E_cond(poly, index, marginal, **kws),
                     marginal, **kws)/(V+(V == 0))*(V != 0)

    return out


def Sens_t_nataf(order, dist, samples, vals, **kws):
    """
    Variance-based decomposition thorugh the Nataf distribution.

    Total order sensitivity indices

    Args:
        order (int):
            Polynomial order used ``orth_ttr``.
        dist (Copula):
            Assumed to be Nataf with independent components
        samples (numpy.ndarray):
            Samples used for evaluation (typically generated from ``dist``.)
        vals (numpy.ndarray):
            Evaluations of the model for given samples.

    Returns:
        (numpy.ndarray) :
            Sensitivity indices with shape ``(len(dist),)+vals.shape[1:]``.
    """
    assert dist.__class__.__name__ == "Copula"
    trans = dist.prm["trans"]
    assert trans.__class__.__name__ == "nataf"
    vals = nump.array(vals)

    cov = trans.prm["C"]
    cov = nump.dot(cov, cov.T)

    marginal = dist.prm["dist"]
    dim = len(dist)

    orth = chaospy.orthogonal.orth_ttr(order, marginal, sort="GR")

    r = range(dim)

    index = [0] + [1]*(dim-1)

    nataf = chaospy.dist.Nataf(marginal, cov, r)
    samples_ = marginal.inv( nataf.fwd( samples ) )
    poly, coeffs = chaospy.collocation.fit_regression(
        orth, samples_, vals, retall=1)

    V = Var(poly, marginal, **kws)

    out = nump.zeros((dim,) + poly.shape)
    out[0] = (V-Var(E_cond(poly, index, marginal, **kws),
                    marginal, **kws))/(V+(V == 0))**(V != 0)

    for i in range(1, dim):

        r = r[1:] + r[:1]
        index = index[-1:] + index[:-1]

        nataf = chaospy.dist.Nataf(marginal, cov, r)
        samples_ = marginal.inv( nataf.fwd( samples ) )
        poly, coeffs = chaospy.collocation.fit_regression(
            orth, samples_, vals, retall=1)

        out[i] = (V-Var(E_cond(poly, index, marginal, **kws),
                        marginal, **kws))/(V+(V == 0))*(V != 0)

    return out


def Sens_nataf(order, dist, samples, vals, **kws):
    """
    Variance-based decomposition through the Nataf distribution.

    Main and total order sensitivity indices

    Args:
        order (int):
            polynomial order used ``orth_ttr``.
        dist (Copula):
            Assumed to be Nataf with independent components
        samples (numpy.ndarray):
            Samples used for evaluation (typically generated from ``dist``.)
        vals (numpy.ndarray):
            Evaluations of the model for given samples.

    Returns:
        (numpy.ndarray):
            Sensitivity indices with shape
            ``kjl``(2, len(dist),)+vals.shape[1:]``. First component is main
            and second is total.
    """
    assert dist.__class__.__name__ == "Copula"
    trans = dist.prm["trans"]
    assert trans.__class__.__name__ == "nataf"
    vals = nump.array(vals)

    cov = trans.prm["C"]
    cov = nump.dot(cov, cov.T)

    marginal = dist.prm["dist"]
    dim = len(dist)

    orth = chaospy.orthogonal.orth_ttr(order, marginal, sort="GR")

    r = range(dim)

    index0 = [0] + [1]*(dim-1)
    index1 = [1] + [0]*(dim-1)

    nataf = chaospy.dist.Nataf(marginal, cov, r)
    samples_ = marginal.inv( nataf.fwd( samples ) )
    poly, coeffs = chaospy.collocation.fit_regression(
        orth, samples_, vals, retall=1)

    V = Var(poly, marginal, **kws)

    out = nump.zeros((2, dim,) + poly.shape)
    out[0, 0] = (V - Var(E_cond(poly, index0, marginal, **kws),
                        marginal, **kws))/(V+(V == 0))**(V != 0)
    out[1, 0] = Var(E_cond(poly, index1, marginal, **kws),
                   marginal, **kws)/(V+(V == 0))*(V != 0)

    for i in range(1, dim):

        r = r[1:] + r[:1]
        index0 = index0[-1:] + index0[:-1]

        nataf = chaospy.dist.Nataf(marginal, cov, r)
        samples_ = marginal.inv( nataf.fwd( samples ) )
        poly, coeffs = chaospy.collocation.fit_regression(
            orth, samples_, vals, retall=1)

        out[0, i] = (V-Var(E_cond(poly, index0, marginal, **kws),
                          marginal, **kws))/(V+(V == 0))*(V != 0)
        out[1, i] = Var(E_cond(poly, index1, marginal, **kws),
                       marginal, **kws)/(V+(V == 0))*(V != 0)

    return out[::-1]
