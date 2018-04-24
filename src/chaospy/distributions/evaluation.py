import logging
import inspect

import networkx
import numpy

from .. import quad


class DependencyError(ValueError):
    """Error that occurs with bad stochastic dependency structures."""


def sorted_dependencies(dist, reverse=False):
    from . import baseclass
    graph = networkx.DiGraph()
    graph.add_node(dist)
    dist_collection = [dist]
    while dist_collection:

        cur_dist = dist_collection.pop()

        values = (value for value in cur_dist.prm.values()
                  if isinstance(value, baseclass.Dist))
        for value in values:
            if value not in graph.node:
                graph.add_node(value)
                dist_collection.append(value)
            graph.add_edge(cur_dist, value)

    if not networkx.is_directed_acyclic_graph(graph):
        raise DependencyError("cycles in dependency structure.")

    sorting = list(networkx.topological_sort(graph))
    if reverse:
        sorting = reversed(sorting)

    for dist in sorting:
        yield dist


def has_argument(key, method):
    try:
        args = inspect.signature(method).parameters
    except AttributeError:
        args = inspect.getargspec(method).args
    return key in args


def get_dependencies(*distributions):
    from . import baseclass
    distributions = [
        set(sorted_dependencies(dist)) for dist in distributions
        if isinstance(dist, baseclass.Dist)
    ]
    if len(distributions) <= 1:
        return set()
    intersections = set.intersection(*distributions)
    return intersections


def load_inputs(
        distribution,
        cache,
        params,
        methodname="",
):
    from . import baseclass
    if cache is None:
        cache = {}
    if params is None:
        params = distribution.prm
    params = params.copy()

    if methodname:

        # self aware and should handle things itself:
        if has_argument("cache", getattr(distribution, methodname)):
            params["cache"] = cache

        # dumb distribution and just wants to evaluate:
        else:
            for key, value in params.items():
                if isinstance(value, baseclass.Dist):
                    if value in cache:
                        params[key] = cache[value]
                    else:
                        raise DependencyError(
                            "under-defined distribution {}.".format(value))

    return cache, params


def evaluate_density(
        distribution,
        x_data,
        cache=None,
        params=None,
):
    logger = logging.getLogger(__name__)
    logger.debug("init evaluate_density: %s", distribution)
    out = numpy.zeros(x_data.shape)
    if hasattr(distribution, "_pdf"):
        cache, params = load_inputs(distribution, cache, params, "_pdf")
        out[:] = distribution._pdf(x_data, **params)

    else:
        from . import approximation
        cache, params = load_inputs(distribution, cache, params)
        out[:] = approximation.approximate_density(
            distribution, x_data, params, cache)

    if distribution in cache:
        out = numpy.where(x_data == cache[distribution], out, 0)
    else:
        cache[distribution] = x_data
    logger.debug("end evaluate_density: %s", distribution)
    return out


def evaluate_forward(
        distribution,
        x_data,
        cache=None,
        params=None,
):
    assert len(x_data) == len(distribution)
    logger = logging.getLogger(__name__)
    logger.debug("init evaluate_forward: %s", distribution)

    cache, params = load_inputs(distribution, cache, params, "_cdf")
    cache[distribution] = x_data
    out = numpy.zeros(x_data.shape)
    out[:] = distribution._cdf(x_data.copy(), **params)
    logger.debug("end evaluate_forward: %s", distribution)
    return out


def evaluate_inverse(
        distribution,
        q_data,
        cache=None,
        params=None
):
    logger = logging.getLogger(__name__)
    logger.debug("init evaluate_inverse: %s", distribution)

    out = numpy.zeros(q_data.shape)
    if hasattr(distribution, "_ppf"):
        cache, params = load_inputs(distribution, cache, params, "_ppf")
        out[:] = distribution._ppf(q_data.copy(), **params)
    else:
        logger.info(
            "distribution %s has no _ppf method; approximating.", distribution)
        from . import approximation
        cache, params = load_inputs(distribution, cache, params, "_cdf")
        out[:] = approximation.approximate_inverse(
            distribution, q_data, cache=cache, params=params)
    cache[distribution] = out
    logger.debug("end evaluate_inverse: %s", distribution)
    return out


def evaluate_bound(
        distribution,
        x_data,
        cache=None,
        params=None,
):
    assert len(x_data) == len(distribution)
    assert len(x_data.shape) == 2
    logger = logging.getLogger(__name__)
    logger.debug("init evaluate_bound: %s", distribution)

    cache, params = load_inputs(distribution, cache, params, "_bnd")
    out = numpy.zeros((2,) + x_data.shape)
    lower, upper = distribution._bnd(x_data.copy(), **params)
    lower = numpy.asfarray(lower)
    upper = numpy.asfarray(upper)

    try:
        out.T[:, :, 0] = lower.T
    except ValueError:
        logger.exception(
            "method %s._bnd returned wrong shape: %s",
            distribution, lower.shape)
        raise
    try:
        out.T[:, :, 1] = upper.T
    except ValueError:
        logger.exception(
            "method %s._bnd returned wrong shape: %s",
            distribution, upper.shape)
        raise

    cache[distribution] = out
    logger.debug("end evaluate_bound: %s", distribution)
    return out


def evaluate_moment(distribution, k_data, cache):
    logger = logging.getLogger(__name__)
    logger.debug("init evaluate_moment: %s", distribution)

    from . import baseclass
    if numpy.all(k_data == 0):
        logger.debug("end evaluate_moment: %s", distribution)
        return 1.
    if (tuple(k_data), distribution) in cache:
        logger.debug("end evaluate_moment: %s", distribution)
        return cache[(tuple(k_data), distribution)]

    params = distribution.prm.copy()

    # self aware and should handle things itself:
    if has_argument("cache", distribution._mom):
        params["cache"] = cache

    # dumb distribution and just wants to evaluate:
    else:
        for key, value in params.items():
            if isinstance(value, baseclass.Dist):
                if (tuple(k_data), value) in cache:
                    params[key] = cache[(tuple(k_data), value)]
                else:
                    raise DependencyError(
                        "under-defined distribution {}.".format(value))

    try:
        out = float(distribution._mom(k_data, **params))
    except DependencyError:
        from . import approximation
        out = approximation.approximate_moment(distribution, k_data)

    cache[(tuple(k_data), distribution)] = out

    logger.debug("end evaluate_moment: %s", distribution)
    return out


def evaluate_recurrence_coefficients(
        distribution,
        k_data,
        cache=None,
        params=None,
):
    logger = logging.getLogger(__name__)
    logger.debug("init evaluate_recurrence_coefficients: %s", distribution)

    from . import baseclass
    cache, params = load_inputs(distribution, cache, params)
    if (tuple(k_data), distribution) in cache:
        logger.debug("end evaluate_recurrence_coefficients: %s", distribution)
        return cache[(tuple(k_data), distribution)]

    # self aware and should handle things itself:
    if has_argument("cache", distribution._ttr):
        params["cache"] = cache

    # dumb distribution and just wants to evaluate:
    else:
        for key, value in params.items():
            if isinstance(value, baseclass.Dist):
                if (tuple(k_data), value) in cache:
                    params[key] = cache[(tuple(k_data), value)]
                else:
                    raise DependencyError(
                        "under-defined distribution {}.".format(value))

    try:
        coeff1, coeff2 = distribution._ttr(k_data, **params)

    except NotImplementedError:
        _, _, coeff1, coeff2 = quad.stieltjes._stieltjes_approx(
            distribution, order=numpy.max(k_data), accuracy=100, normed=False)
        range_ = numpy.arange(len(distribution), dtype=int)
        coeff1 = coeff1[range_, k_data]
        coeff2 = coeff2[range_, k_data]

    out = numpy.zeros((2,) + k_data.shape)
    out.T[:, 0] = numpy.asarray(coeff1).T
    out.T[:, 1] = numpy.asarray(coeff2).T
    if len(distribution) == 1:
        out = out[:, 0]

    logger.debug("end evaluate_recurrence_coefficients: %s", distribution)
    return out
