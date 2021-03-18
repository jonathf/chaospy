"""Collection of approximation functions."""
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
        xloc0=None,
        iterations=300,
        tolerance=1e-12,
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
        >>> inverse = distribution.inv(qloc)
        >>> inverse.round(4)
        array([ 987.1845,  991.5838, 1012.8155])
        >>> distribution._ppf = None
        >>> numpy.allclose(
        ...     approximate_inverse(distribution, 0, qloc), inverse)
        True

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
        xlower = numpy.broadcast_to(bounds[0], qloc.shape)
        xupper = numpy.broadcast_to(bounds[1], qloc.shape)
        _lower, distribution._lower = distribution._lower, lambda **kws: xlower
        _upper, distribution._upper = distribution._upper, lambda **kws: xupper

    xloc = xlower+qloc*(xlower+xupper) if xloc0 is None else xloc0
    uloc = numpy.zeros(qloc.shape)
    ulower = -qloc
    uupper = 1-qloc
    indices = numpy.ones(qloc.shape[-1], dtype=bool)

    cache_copy = cache.copy()
    parameters_copy = distribution._parameters.copy()
    if parameters is not None:

        assert not any([isinstance(value, chaospy.Distribution)
                        for value in parameters.values()])
        distribution._parameters.update(parameters)

    for idx_ in range(2*iterations):

        cache.clear()
        cache.update(cache_copy)
        # evaluate function:
        uloc = numpy.where(
            indices, distribution._get_fwd(xloc, idx, cache)-qloc, uloc)

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
            cache.clear()
            cache.update(cache_copy)
            try:
                derivative = distribution._get_pdf(xloc, idx, cache)
            except chaospy.UnsupportedFeature:
                cache.clear()
                cache.update(cache_copy)
                derivative = approximate_density(distribution, idx, xloc, cache)
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
        # print("Too many iterations required to estimate inverse.")
        # print("%d out of %d did not converge." % (numpy.sum(indices), len(indices)))

    cache.clear()
    cache.update(cache_copy)
    distribution._parameters.clear()
    distribution._parameters.update(parameters_copy)
    if bounds is not None:
        distribution._lower = _lower
        distribution._upper = _upper
    logger.debug("%s: ppf approx used %d steps", distribution, idx_/2)
    # print("%s: ppf approx used %d steps" % (distribution, idx_/2))
    return xloc

MOMENTS_QUADS = {}


def approximate_moment(
        distribution,
        k_loc,
        order=100000,
        rule="clenshaw_curtis",
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
            The quadrature order used in approximation.
        rule (str):
            Quadrature rule for integrating moments.
        kwargs:
            Extra args passed to :func:`chaospy.generate_quadrature`.

    Examples:
        >>> distribution = chaospy.Uniform(1, 4)
        >>> round(chaospy.approximate_moment(distribution, (1,)), 4)
        2.5
        >>> round(chaospy.approximate_moment(distribution, (2,)), 4)
        7.0

    """
    assert isinstance(distribution, chaospy.Distribution)
    k_loc = numpy.asarray(k_loc)
    if len(distribution) > 1:
        assert not distribution.stochastic_dependent, (
            "Dependent distributions does not support moment approximation.")
        assert len(k_loc) == len(distribution), "incorrect size of exponents"
        return numpy.prod([
            approximate_moment(distribution[idx], (k_loc[idx],),
                               order=order, rule=rule, **kwargs)
            for idx in range(len(distribution))
        ], axis=0)

    k_loc = tuple(k_loc.tolist())
    assert all([isinstance(k, int) for k in k_loc]), (
        "exponents must be integers: %s found" % type(k_loc[0]))

    order = int(1e5 if order is None else order)
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
        step_size=1e-7
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
        step_size (float):
            The relative step size between two points used to calculate the
            derivative.

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
    cache = {} if cache is None else cache
    assert (idx, distribution) not in cache
    xloc = numpy.asfarray(xloc)
    assert xloc.ndim == 1
    lower, upper = numpy.min(xloc), numpy.max(xloc)
    middle = .5*(lower+upper)
    step_size = (numpy.where(xloc < middle, step_size, -step_size)*
                 numpy.clip(numpy.abs(xloc), 1, None))

    cache1 = cache.copy()
    floc1 = distribution._get_fwd(xloc, idx, cache=cache1)
    cache2 = cache.copy()
    floc2 = distribution._get_fwd(xloc+step_size, idx, cache=cache2)
    floc = numpy.abs((floc2-floc1)/step_size)

    # weave a history of pdf from two cdf streams
    for key in set(cache1).difference(cache):
        cache[key] = cache1[key][0], ((cache2[key][1]-cache1[key][1])/
                                      (cache2[key][0]-cache1[key][0]))


    assert floc.shape == xloc.shape
    return floc
