import logging

import numpy
import chaospy


def approximate_inverse(
        distribution,
        idx,
        qloc,
        bounds=None,
        cache=None,
        parameters=None,
        iterations=300,
        tolerance=1e-5,
):
    """
    Calculate the approximation of the inverse Rosenblatt transformation.

    Uses a hybrid Newton-Raphson and binary search method to converge to the
    inverse values. Includes forward Rosenblatt transformations, probability
    density function (its derivative), and if not provided, boundary function.

    Args:
        distribution (Distribution):
            Distribution to estimate inverse Rosenblatt on.
        idx (int):
            The dimension to take approximation along.
        qloc (numpy.ndarray):
            Input values. All values must be on unit interval ``(0, 1)`` and
            ``qloc.shape == (dim,size)`` where dim is the number of dimensions
            in distribution and size is the number of values to calculate
            simultaneously.
        bounds (Optional[Tuple[numpy.ndarray, numpy.ndarray]]):
            Assuming lower and upper bounds is not available, this provides
            outer bounds for lower and upper to use instead.
        cache (Optional[Dict[Distribution, numpy.ndarray]]):
            Memory cache for the location in the evaluation so far.
        parameters (Optional[Dict[str, Any]]):
            The parameters to use. If omitted, get the parameters from
            distribution.
        iterations (int):
            The maximum number of iterations allowed.
        tolerance (float):
            Tolerance criterion determining convergence.

    Returns:
        (numpy.ndarray):
            Approximation of inverse Rosenblatt transformation.

    Example:
        >>> distribution = chaospy.Normal(1000, 10)
        >>> qloc = numpy.array([0.1, 0.2, 0.9])
        >>> approximate_inverse(distribution, 0, qloc).round(4)
        array([ 987.1845,  991.5839, 1012.8153])
        >>> distribution.inv(qloc).round(4)
        array([ 987.1845,  991.5838, 1012.8155])

    """
    logger = logging.getLogger(__name__)
    logger.debug("init approximate_inverse: %s", distribution)
    logger.debug("cache: %s", cache)

    # lots of initial values:
    if cache is None:
        cache = {}
    if bounds is None:
        xlower = distribution._get_lower(idx, cache.copy())
        xupper = distribution._get_upper(idx, cache.copy())
    else:
        xlower, xupper = bounds
    xlower = numpy.broadcast_to(xlower, qloc.shape)
    xupper = numpy.broadcast_to(xupper, qloc.shape)
    xloc = 0.5*(xlower+xupper)
    uloc = numpy.zeros(qloc.shape)
    ulower = -qloc
    uupper = 1-qloc
    indices = numpy.ones(qloc.shape[-1], dtype=bool)

    if parameters is None:
        parameters = distribution.get_parameters(idx, cache, assert_numerical=True)
    else:
        assert not any([isinstance(value, chaospy.Distribution)
                        for value in parameters.values()])
        for name, param in parameters.items():
            if isinstance(distribution._parameters[name], chaospy.Distribution):
                cache[(idx, distribution._parameters[name])] = param

    for idx_ in range(2*iterations):

        # evaluate function:
        uloc = numpy.where(
            indices, distribution._cdf(xloc, **parameters)-qloc, uloc)

        # convergence criteria:
        indices &= numpy.abs(uloc) > tolerance
        if not numpy.any(indices):
            break

        # narrow down boundaries:
        ulower = numpy.where(indices & (uloc < 0), uloc, ulower)
        xlower = numpy.where(indices & (uloc < 0), xloc, xlower)
        uupper = numpy.where(indices & (uloc > 0), uloc, uupper)
        xupper = numpy.where(indices & (uloc > 0), xloc, xupper)

        # Newton increment every second iteration:
        xloc_ = numpy.inf
        if idx_ % 2 == 0:
            derivative = distribution._pdf(xloc, **parameters)
            derivative = numpy.where(derivative, derivative, 1)
            xloc_ = xloc-uloc/derivative

        # use binary search if Newton increment is outside bounds:
        weight = numpy.random.random()
        xloc_ = numpy.where((xloc_ < xupper) & (xloc_ > xlower),
                            xloc_, weight*xupper+(1-weight)*xlower)
        xloc = numpy.where(indices, xloc_, xloc)

    else:
        logger.warning(
            "Too many iterations required to estimate inverse.")
        logger.info("%d out of %d did not converge.",
            numpy.sum(indices), len(indices))

    logger.debug("%s: ppf approx used %d steps", distribution, idx_/2)
    return xloc

MOMENTS_QUADS = {}


def approximate_moment(
        distribution,
        k_loc,
        order=None,
        rule="fejer",
        **kwargs
):
    """
    Approximation method for estimation of raw statistical moments.

    Uses quadrature integration to estimate the values.

    Args:
        distribution (Distribution):
            Distribution domain with dim=len(distribution)
        k_loc (Sequence[int, ...]):
            The exponents of the moments of interest with ``shape == (dim,)``.
        order (int):
            The quadrature order used in approximation. If omitted, calculated
            to be ``1000/log2(len(distribution)+1)``.
        rule (str):
            Quadrature rule for integrating moments.
        kwargs:
            Extra args passed to `chaospy.generate_quadrature`.

    Examples:
        >>> distribution = chaospy.Uniform(1, 4)
        >>> round(chaospy.approximate_moment(distribution, (1,)), 4)
        2.5
        >>> round(chaospy.approximate_moment(distribution, (2,)), 4)
        7.0

    """
    if order is None:
        order = int(1000./numpy.log2(len(distribution)+1))
    assert isinstance(order, int)
    assert isinstance(distribution, chaospy.Distribution)
    k_loc = tuple(numpy.asarray(k_loc).tolist())
    assert len(k_loc) == len(distribution), "incorrect size of exponents"
    assert all([isinstance(k, int) for k in k_loc]), (
        "exponents must be integers: %s found" % type(k_loc[0]))

    if (distribution, order) not in MOMENTS_QUADS:
        MOMENTS_QUADS[distribution, order] = chaospy.generate_quadrature(
            order, distribution, rule=rule, **kwargs)
    X, W = MOMENTS_QUADS[distribution, order]

    if k_loc in distribution._mom_cache:
        return distribution._mom_cache[k_loc]

    out = float(numpy.sum(numpy.prod(X.T**k_loc, 1)*W))
    distribution._mom_cache[k_loc] = out
    return out


def approximate_density(
        distribution,
        idx,
        xloc,
        cache=None,
        tolerance=1e-7
):
    """
    Approximate the probability density function.

    Args:
        distribution (Distribution):
            Distribution in question. May not be an advanced variable.
        idx (int):
            The dimension to take approximation along.
        xloc (numpy.ndarray):
            Location coordinates. Requires that xloc.shape=(len(distribution), K).
        cache (Optional[Dict[Distribution, Tuple[numpy.ndarray, numpy.ndarray]]]):
            Current state in the evaluation graph. If omitted, assume that
            evaluations should be done from scratch.
        tolerance (float):
            Acceptable error level for the approximations

    Returns:
        numpy.ndarray: Local probability density function with
            ``out.shape == xloc.shape``. To calculate actual density function,
            evaluate ``numpy.prod(out, 0)``.

    Example:
        >>> distribution = chaospy.Normal(1000, 10)
        >>> xloc = numpy.array([990, 1000, 1010])
        >>> approximate_density(distribution, 0, xloc).round(4)
        array([0.0242, 0.0399, 0.0242])
        >>> distribution.pdf(xloc).round(4)
        array([0.0242, 0.0399, 0.0242])

    """
    cache = cache or {}
    xloc = numpy.asfarray(xloc)
    assert xloc.ndim == 1
    lo, up = numpy.min(xloc), numpy.max(xloc)
    mu = .5*(lo+up)
    tolerance = (numpy.where(xloc < mu, tolerance, -tolerance)*
                 numpy.clip(numpy.abs(xloc), 1, None))

    floc1 = distribution._get_fwd(xloc, idx, cache=cache.copy())
    floc2 = distribution._get_fwd(xloc+tolerance, idx, cache=cache.copy())
    floc = numpy.abs((floc2-floc1)/tolerance)

    assert floc.shape == xloc.shape
    return floc
