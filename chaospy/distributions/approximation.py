import logging
from functools import partial

import numpy
import chaospy

from . import evaluation


def approximate_inverse(
        distribution,
        qloc,
        parameters=None,
        cache=None,
        iterations=300,
        tol=1e-5,
        seed=None,
):
    """
    Calculate the approximation of the inverse Rosenblatt transformation.

    Uses forward Rosenblatt, probability density function (derivative) and
    boundary function to apply hybrid Newton-Raphson and binary search method.

    Args:
        distribution (Dist):
            Distribution to estimate inverse Rosenblatt.
        qloc (numpy.ndarray):
            Input values. All values must be on [0,1] and
            ``qloc.shape == (dim,size)`` where dim is the number of dimensions
            in distribution and size is the number of values to calculate
            simultaneously.
        parameters (Optional[Dict[Dist, numpy.ndarray]]):
            Parameters for the distribution.
        cache (Optional[Dict[Dist, numpy.ndarray]]):
            Memory cache for the location in the evaluation so far.
        iterations (int):
            The number of iterations allowed to be performed
        tol (float):
            Tolerance parameter determining convergence.
        seed (Optional[int]):
            Fix random seed.

    Returns:
        (numpy.ndarray):
            Approximation of inverse Rosenblatt transformation.

    Example:
        >>> distribution = chaospy.Normal(1000, 10)
        >>> qloc = numpy.array([[0.1, 0.2, 0.9]])
        >>> approximate_inverse(distribution, qloc, seed=1234).round(4)
        array([[ 987.1846,  991.5839, 1012.8152]])
        >>> distribution.inv(qloc).round(4)
        array([[ 987.1845,  991.5838, 1012.8155]])
    """
    logger = logging.getLogger(__name__)
    logger.debug("init approximate_inverse: %s", distribution)

    # lots of initial values:
    if cache is None:
        cache = {}
    xlower = distribution.lower
    xupper = distribution.upper
    xloc = 0.5*(xlower+xupper)
    xloc = (xloc.T + numpy.zeros(qloc.shape).T).T
    xlower = (xlower.T + numpy.zeros(qloc.shape).T).T
    xupper = (xupper.T + numpy.zeros(qloc.shape).T).T
    uloc = numpy.zeros(qloc.shape)
    ulower = -qloc
    uupper = 1-qloc

    for dim in distribution._precedence_order():
        indices = numpy.ones(qloc.shape[-1], dtype=bool)

        for idx in range(2*iterations):

            # evaluate function:
            uloc[dim, indices] = (evaluation.evaluate_forward(
                distribution, xloc, cache=cache.copy(),
                parameters=parameters)-qloc)[dim, indices]

            # convergence criteria:
            indices[indices] = numpy.any(numpy.abs(uloc) > tol, 0)[indices]
            if not numpy.any(indices):
                break

            # narrow down lower boundary:
            ulower[dim, indices] = numpy.where(uloc < 0, uloc, ulower)[dim, indices]
            xlower[dim, indices] = numpy.where(uloc < 0, xloc, xlower)[dim, indices]

            # narrow down upper boundary:
            uupper[dim, indices] = numpy.where(uloc > 0, uloc, uupper)[dim, indices]
            xupper[dim, indices] = numpy.where(uloc > 0, xloc, xupper)[dim, indices]

            # Newton increment every second iteration:
            xloc_ = numpy.inf
            if idx % 2 == 0:
                derivative = evaluation.evaluate_density(
                    distribution, xloc, cache=cache.copy(),
                    parameters=parameters)[dim, indices]
                derivative = numpy.where(derivative, derivative, numpy.inf)

                xloc_ = xloc[dim, indices] - uloc[dim, indices] / derivative

            # use binary search if Newton increment is outside bounds:
            weight = numpy.random.random()
            xloc[dim, indices] = numpy.where(
                (xloc_ < xupper[dim, indices]) & (xloc_ > xlower[dim, indices]),
                xloc_, (weight*xupper+(1-weight)*xlower)[dim, indices])

        else:
            logger.warning(
                "Too many iterations (dim %d) required to estimate inverse.", dim)
            logger.info("%d out of %d did not converge.",
                numpy.sum(indices), len(indices))

    logger.debug("end approximate_inverse: %s", distribution)
    return xloc

MOMENTS_QUADS = {}
MOMENTS_RESULTS = {}


def approximate_moment(
        dist,
        k_loc,
        order=None,
        rule="fejer",
        **kws
):
    """
    Approximation method for estimation of raw statistical moments.

    Args:
        dist (Dist):
            Distribution domain with dim=len(dist)
        k_loc (Sequence[int, ...]):
            The exponents of the moments of interest with shape (dim,).
        order (int):
            The quadrature order used in approximation. If omitted, calculated
            to be ``1000/log2(len(dist)+1)``.
        rule (str):
            Quadrature rule for integrating moments.
        kws:
            Extra args passed to `chaospy.generate_quadrature`.
    """
    if order is None:
        order = int(1000./numpy.log2(len(dist)+1))
    assert isinstance(order, int)
    assert isinstance(dist, chaospy.Dist)
    k_loc = numpy.asarray(k_loc, dtype=int)
    assert k_loc.shape == (len(dist),), "incorrect size of exponents"
    assert k_loc.dtype == int, "exponents have the wrong dtype"

    if (tuple(k_loc), dist) in MOMENTS_RESULTS:
        return MOMENTS_RESULTS[tuple(k_loc), dist]

    if (tuple(k_loc), dist, order) not in MOMENTS_QUADS:
        MOMENTS_QUADS[tuple(k_loc), dist, order] = chaospy.generate_quadrature(
            order, dist, rule=rule, **kws)
    X, W = MOMENTS_QUADS[tuple(k_loc), dist, order]

    out = numpy.sum(numpy.prod(X.T**k_loc, 1)*W)
    MOMENTS_RESULTS[tuple(k_loc), dist] = out
    return out


def approximate_density(
        dist,
        xloc,
        parameters=None,
        cache=None,
        eps=1.e-7
):
    """
    Approximate the probability density function.

    Args:
        dist : Dist
            Distribution in question. May not be an advanced variable.
        xloc : numpy.ndarray
            Location coordinates. Requires that xloc.shape=(len(dist), K).
        eps : float
            Acceptable error level for the approximations
        retall : bool
            If True return Graph with the next calculation state with the
            approximation.

    Returns:
        numpy.ndarray: Local probability density function with
            ``out.shape == xloc.shape``. To calculate actual density function,
            evaluate ``numpy.prod(out, 0)``.

    Example:
        >>> distribution = chaospy.Normal(1000, 10)
        >>> xloc = numpy.array([[990, 1000, 1010]])
        >>> approximate_density(distribution, xloc).round(4)
        array([[0.0242, 0.0399, 0.0242]])
        >>> distribution.pdf(xloc).round(4)
        array([[0.0242, 0.0399, 0.0242]])
    """
    if parameters is None:
        parameters = dist.prm.copy()
    if cache is None:
        cache = {}

    xloc = numpy.asfarray(xloc)
    lo, up = numpy.min(xloc), numpy.max(xloc)
    mu = .5*(lo+up)
    eps = numpy.where(xloc < mu, eps, -eps)*xloc

    floc = evaluation.evaluate_forward(
        dist, xloc, parameters=parameters.copy(), cache=cache.copy())
    for d in range(len(dist)):
        xloc[d] += eps[d]
        tmp = evaluation.evaluate_forward(
            dist, xloc, parameters=parameters.copy(), cache=cache.copy())
        floc[d] -= tmp[d]
        xloc[d] -= eps[d]

    floc = numpy.abs(floc / eps)
    return floc
