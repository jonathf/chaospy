import logging
from functools import partial
import numpy

from . import evaluation
from .. import quad


def find_interior_point(
        distribution,
        params=None,
        cache=None,
        iterations=1000,
        retall=False,
        seed=None,
):
    """
    Find interior point of the distribution where forward evaluation is
    guarantied to be both ``distribution.fwd(xloc) > 0`` and
    ``distribution.fwd(xloc) < 1``.

    Args:
        distribution (Dist): Distribution to find interior on.
        params (Optional[Dict[Dist, numpy.ndarray]]): Parameters for the
            distribution.
        cache (Optional[Dict[Dist, numpy.ndarray]]): Memory cache for the
            location in the evaluation so far.
        iterations (int): The number of iterations allowed to be performed
        retall (bool): If provided, lower and upper bound which guaranties that
            ``distribution.fwd(lower) == 0`` and
            ``distribution.fwd(upper) == 1`` is returned as well.
        seed (Optional[int]): Fix random seed.

    Returns:
        numpy.ndarray: An input array with shape ``(len(distribution),)`` which
        is guarantied to be on the interior of the probability distribution.


    Example:
        >>> distribution = chaospy.MvNormal([1, 2, 3], numpy.eye(3)+.03)
        >>> midpoint, lower, upper = find_interior_point(
        ...     distribution, retall=True, seed=1234)
        >>> print(lower.T)
        [[-64. -64. -64.]]
        >>> print(numpy.around(midpoint, 4).T)
        [[  0.6784 -33.7687 -19.0182]]
        >>> print(upper.T)
        [[16. 16. 16.]]
        >>> distribution = chaospy.Uniform(1000, 1010)
        >>> midpoint, lower, upper = find_interior_point(
        ...     distribution, retall=True, seed=1234)
        >>> print(lower, numpy.around(midpoint, 4), upper)
        [[-1.]] [[1009.8873]] [[1024.]]
    """
    random_state = numpy.random.get_state()
    numpy.random.seed(seed)

    forward = partial(evaluation.evaluate_forward, cache=cache,
                      distribution=distribution, params=params)

    dim = len(distribution)
    upper = numpy.ones((dim, 1))
    for _ in range(100):
        indices = forward(x_data=upper) < 1
        if not numpy.any(indices):
            break
        upper[indices] *= 2

    lower = -numpy.ones((dim, 1))
    for _ in range(100):
        indices = forward(x_data=lower) > 0
        if not numpy.any(indices):
            break
        lower[indices] *= 2

    for _ in range(iterations):

        rand = numpy.random.random(dim)
        proposal = (rand*lower.T + (1-rand)*upper.T).T
        evals = forward(x_data=proposal)

        indices0 = evals > 0
        indices1 = evals < 1

        range_ = numpy.random.choice(dim, size=dim, replace=False)

        upper_ = numpy.where(indices1, upper, evals)
        for idx in range_:
            if upper.flatten()[idx] == upper_.flatten()[idx]:
                continue
            if numpy.all(forward(x_data=upper_) == 1):
                upper = upper_
                break
            upper_[idx] = upper[idx]

        lower_ = numpy.where(indices0, lower, evals)
        for idx in range_:
            if lower.flatten()[idx] == lower_.flatten()[idx]:
                continue
            if numpy.all(forward(x_data=lower_) == 0):
                lower = lower_
                break
            lower_[idx] = lower[idx]

        if numpy.all(indices0 & indices1):
            break

    else:
        if retall:
            return proposal, lower, upper
        return proposal
        raise evaluation.DependencyError(
            "Too many iterations required to find interior point.")

    numpy.random.set_state(random_state)
    if retall:
        return proposal, lower, upper
    return proposal


def approximate_inverse(
        distribution,
        qloc,
        params=None,
        cache=None,
        iterations=100,
        tol=1e-5,
        seed=None,
):
    """
    Calculate the approximation of the inverse Rosenblatt transformation.

    Uses forward Rosenblatt, probability density function (derivative) and
    boundary function to apply hybrid Newton-Raphson and binary search method.

    Args:
        distribution (Dist): Distribution to estimate inverse Rosenblatt.
        qloc (numpy.ndarray): Input values. All values must be on [0,1] and
            ``qloc.shape == (dim,size)`` where dim is the number of dimensions
            in distribution and size is the number of values to calculate
            simultaneously.
        params (Optional[Dict[Dist, numpy.ndarray]]): Parameters for the
            distribution.
        cache (Optional[Dict[Dist, numpy.ndarray]]): Memory cache for the
            location in the evaluation so far.
        iterations (int): The number of iterations allowed to be performed
        tol (float): Tolerance parameter determining convergence.
        seed (Optional[int]): Fix random seed.

    Returns:
        numpy.ndarray: Approximation of inverse Rosenblatt transformation.

    Example:
        >>> distribution = chaospy.Normal(1000, 10)
        >>> qloc = numpy.array([[0.1, 0.2, 0.9]])
        >>> print(numpy.around(approximate_inverse(
        ...     distribution, qloc, seed=1234), 4))
        [[ 987.1845  991.5838 1012.8155]]
        >>> print(numpy.around(distribution.inv(qloc), 4))
        [[ 987.1845  991.5838 1012.8155]]
    """
    logger = logging.getLogger(__name__)
    logger.debug("init approximate_inverse: %s", distribution)

    # lots of initial values:
    xloc, xlower, xupper = find_interior_point(
        distribution, cache=cache, params=params, retall=True, seed=seed)
    xloc = (xloc.T * numpy.zeros(qloc.shape).T).T
    xlower = (xlower.T + numpy.zeros(qloc.shape).T).T
    xupper = (xupper.T + numpy.zeros(qloc.shape).T).T
    uloc = numpy.zeros(qloc.shape)
    ulower = -qloc
    uupper = 1-qloc
    indices = numpy.ones(qloc.shape[-1], dtype=bool)

    for idx in range(2*iterations):

        # evaluate function:
        uloc[:, indices] = (evaluation.evaluate_forward(
            distribution, xloc, cache=cache, params=params)-qloc)[:, indices]

        # convergence criteria:
        indices[indices] = numpy.any(numpy.abs(xupper-xlower) > tol, 0)[indices]
        logger.debug(
            "iter: %s : %s : %s (%s)",
            numpy.mean(xlower, -1),
            numpy.mean(xloc, -1),
            numpy.mean(xupper, -1),
            numpy.mean(indices),
        )
        if not numpy.any(indices):
            break

        # narrow down lower boundary:
        ulower[:, indices] = numpy.where(uloc <= 0, uloc, ulower)[:, indices]
        xlower[:, indices] = numpy.where(uloc <= 0, xloc, xlower)[:, indices]

        # narrow down upper boundary:
        uupper[:, indices] = numpy.where(uloc >= 0, uloc, uupper)[:, indices]
        xupper[:, indices] = numpy.where(uloc >= 0, xloc, xupper)[:, indices]

        # Newton increment every second iteration:
        xloc_ = numpy.inf
        if idx % 2 == 0:
            derivative = evaluation.evaluate_density(
                distribution, xloc, cache=cache, params=params)[:, indices]
            derivative = numpy.where(derivative, derivative, numpy.inf)

            xloc_ = xloc[:, indices] - uloc[:, indices] / derivative

        # use binary search if Newton increment is outside bounds:
        xloc[:, indices] = numpy.where(
            (xloc_ < xupper[:, indices]) & (xloc_ > xlower[:, indices]),
            xloc_, 0.5*(xupper+xlower)[:, indices])

    else:
        logger.warning(
            "Too many iterations required to estimate inverse.")
        logger.info("{} out of {} did not converge.".format(
            numpy.sum(indices), len(indices)))
    logger.debug("end approximate_inverse: %s", distribution)
    return xloc



def approximate_moment(
        dist,
        K,
        retall=False,
        control_var=None,
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
        acc : int, optional
            The order of quadrature/MCI
        sparse : bool
            If True used Smolyak's sparse grid instead of normal tensor
            product grid in numerical integration.
        rule : str
            Quadrature rule
            Key     Description
            ----    -----------
            "G"     Optiomal Gaussian quadrature from Golub-Welsch
                    Slow for high order and composit is ignored.
            "E"     Gauss-Legendre quadrature
            "C"     Clenshaw-Curtis quadrature. Exponential growth rule is
                    used when sparse is True to make the rule nested.

            Monte Carlo Integration
            Key     Description
            ----    -----------
            "H"     Halton sequence
            "K"     Korobov set
            "L"     Latin hypercube sampling
            "M"     Hammersley sequence
            "R"     (Pseudo-)Random sampling
            "S"     Sobol sequence

        composit : int, array_like optional
            If provided, composit quadrature will be used.
            Ignored in the case if gaussian=True.

            If int provided, determines number of even domain splits
            If array of ints, determines number of even domain splits along
                each axis
            If array of arrays/floats, determines location of splits

        antithetic : array_like, optional
            List of bool. Represents the axes to mirror using antithetic
            variable during MCI.
    """
    dim = len(dist)
    shape = K.shape
    size = int(K.size/dim)
    K = K.reshape(dim, size)

    if dim > 1:
        shape = shape[1:]

    order = kws.pop("order", 40)
    X, W = quad.generate_quadrature(order, dist, **kws)

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
        params=None,
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
        >>> print(numpy.around(approximate_density(distribution, xloc), 4))
        [[0.0242 0.0399 0.0242]]
        >>> print(numpy.around(distribution.pdf(xloc), 4))
        [[0.0242 0.0399 0.0242]]
    """
    if params is None:
        params = dist.prm.copy()
    if cache is None:
        cache = {}

    xloc = numpy.asfarray(xloc)
    lo, up = numpy.min(xloc), numpy.max(xloc)
    mu = .5*(lo+up)
    eps = numpy.where(xloc < mu, eps, -eps)*xloc

    floc = evaluation.evaluate_forward(
        dist, xloc, params=params.copy(), cache=cache.copy())
    for d in range(len(dist)):
        xloc[d] += eps[d]
        tmp = evaluation.evaluate_forward(
            dist, xloc, params=params.copy(), cache=cache.copy())
        floc[d] -= tmp[d]
        xloc[d] -= eps[d]

    floc = numpy.abs(floc / eps)
    return floc
