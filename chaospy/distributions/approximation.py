import logging
from functools import partial
import numpy

from . import evaluation
from .. import quadrature


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


def approximate_moment(
        dist,
        K,
        retall=False,
        control_var=None,
        rule="fejer",
        order=1000,
        **kws
):
    """
    Approximation method for estimation of raw statistical moments.

    Args:
        dist : Dist
            Distribution domain with dim=len(dist)
        K : numpy.ndarray
            The exponents of the moments of interest with shape (dim,K).
        control_var : Dist
            If provided will be used as a control variable to try to reduce
            the error.
        acc (:py:data:typing.Optional[int]):
            The order of quadrature/MCI
        sparse : bool
            If True used Smolyak's sparse grid instead of normal tensor
            product grid in numerical integration.
        rule : str
            Quadrature rule.
        antithetic (:py:data:typing.Optional[numpy.ndarray]):
            List of bool. Represents the axes to mirror using antithetic
            variable during MCI.
    """
    dim = len(dist)
    shape = K.shape
    size = int(K.size/dim)
    K = K.reshape(dim, size)

    if dim > 1:
        shape = shape[1:]

    X, W = quadrature.generate_quadrature(order, dist, rule=rule, **kws)

    grid = numpy.mgrid[:len(X[0]), :size]
    X = X.T[grid[0]].T
    K = K.T[grid[1]].T
    out = numpy.prod(X**K, 0)*W

    if control_var is not None:

        Y = control_var.ppf(dist.fwd(X))
        mu = control_var.mom(numpy.eye(len(control_var)))

        if (mu.size == 1) and (dim > 1):
            mu = mu.repeat(dim)

        for d in range(dim):
            alpha = numpy.cov(out, Y[d])[0, 1]/numpy.var(Y[d])
            out -= alpha*(Y[d]-mu)

    out = numpy.sum(out, -1)
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
