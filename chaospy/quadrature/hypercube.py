from functools import partial

import numpy
import chaospy

from .utils import combine_quadrature


def hypercube_quadrature(
    quad_func,
    order,
    domain,
    segments=None,
    auto_scale=True,
):
    """
    Enhance simple 1-dimensional unit quadrature with extra features.

    These features include handling of:

    * Distribution as domain by embedding density into the weights
    * Scale to any intervals
    * Multivariate support
    * Repeat quadrature into segments

    Args:
        quad_func (Callable):
            Function that creates quadrature abscissas and weights. If
            ``auto_scale`` is true, the function should be on the form:
            ``abscissas, weights = quad_func(order)`` and be defined on the unit
            interval. Otherwise the call signature should be
            ``abscissas, weights = quad_func(order, lower, upper)`` and it
            should be defined on interval bound by ``lower`` and ``upper``.
        order (int, Sequence[int]):
            The quadrature order passed to the quadrature function.
        domain (Tuple[float, float], :class:`chaospy.Distribution`):
            Either interval on the format ``(lower, upper)`` or a distribution
            to integrate over. If the latter, weights are adjusted to
            incorporate the density at abscissas.
        segments (Optional[int], Sequence[float]):
            The number segments to split the interval on. If sequence is
            provided, use as segment edges instead.
        kwargs (Any):
            Extra keyword arguments passed to `quad_func`.

    Returns:
        Same as ``quad_func`` but adjusted to incorporate extra features.

    Examples:
        >>> def my_quad(order):
        ...     return (numpy.linspace(0, 1, order+1),
        ...             1./numpy.full(order+1, order+2))
        >>> my_quad(2)
        (array([0. , 0.5, 1. ]), array([0.25, 0.25, 0.25]))
        >>> hypercube_quadrature(my_quad, 2, domain=chaospy.Uniform(-1, 1))
        (array([[-1.,  0.,  1.]]), array([0.25, 0.25, 0.25]))
        >>> abscissas, weights = hypercube_quadrature(my_quad, (1, 1), domain=(0, 1))
        >>> abscissas
        array([[0., 0., 1., 1.],
               [0., 1., 0., 1.]])
        >>> weights.round(5)
        array([0.11111, 0.11111, 0.11111, 0.11111])

    """
    if segments is None:
        order, domain = align_arguments(order, domain)
        kwargs = dict(order=order)
    else:
        order, domain, segments = align_arguments(order, domain, segments)
        kwargs = dict(order=order, segments=segments)

    quad_func = partial(ensure_output, quad_func=quad_func)
    if auto_scale:
        if segments is not None:
            quad_func = partial(split_into_segments, quad_func=quad_func)
        quad_func = partial(scale_quadrature, quad_func=quad_func)
    quad_func = partial(univariate_to_multivariate, quad_func=quad_func)
    if isinstance(domain, chaospy.Distribution):
        quad_func = partial(
            distribution_to_domain, quad_func=quad_func, distribution=domain)
    else:
        quad_func = partial(
            quad_func, lower=numpy.asarray(domain[0]), upper=numpy.asarray(domain[1]))

    return quad_func(**kwargs)


def align_arguments(
    order,
    domain,
    segments=None,
):
    """
    Extract dimensions from input arguments and broadcast relevant parts.

    Args:
        order (int, Sequence[int]):
            The quadrature order passed to the quadrature function.
        domain (Tuple[float, float], :func:`chaospy.Distribution`):
            Either interval on the format ``(lower, upper)`` or a distribution
            to integrate over.
        segments (Optional[int], Sequence[float]):
            The number segments to split the interval on. If sequence is
            provided, use as segment edges instead.

    Examples:
        >>> order, domain, segments = align_arguments(1, chaospy.Uniform(0, 1), 1)
        >>> order, domain, segments
        (array([1]), Uniform(), array([1]))
        >>> distribution = chaospy.Iid(chaospy.Uniform(0, 1), 2)
        >>> order, domain = align_arguments(1, distribution)
        >>> order, domain
        (array([1, 1]), Iid(Uniform(), 2))

    """
    args = [numpy.asarray(order)]
    if isinstance(domain, chaospy.Distribution):
        args += [numpy.zeros(len(domain))]
    else:
        args += list(domain)
    if segments is not None:
        segments = numpy.atleast_1d(segments)
        assert segments.ndim <= 2
        if segments.ndim == 2:
            args.append(segments[:, 0])
        else:
            args.append(segments)
    args = numpy.broadcast_arrays(*args)

    output = [args.pop(0)]
    if not isinstance(domain, chaospy.Distribution):
        output += [(args.pop(0), args.pop(0))]
    else:
        output += [domain]
    if segments is not None:
        if segments.ndim == 2:
            segments = numpy.broadcast_arrays(segments, order)[0].T
        else:
            segments = args.pop(-1)
        output += [segments]
    return tuple(output)


def ensure_output(quad_func, **kwargs):
    """
    Converts arrays to python native types and ensure quadrature output sizes.

    Args:
        quad_func (Callable):
            Function that creates quadrature abscissas and weights.
        kwargs (Any):
            Extra keyword arguments passed to `quad_func`.

    Returns:
        Same as ``quad_func(order, **kwargs)`` except numpy elements in
        ``kwargs`` is replaced with Python native counterparts and
        ``abscissas`` is ensured to be at least 2-dimensional.

    Examples:
        >>> def my_quad(order):
        ...     return (numpy.linspace(0, 1, order+1),
        ...             1./numpy.full(order+1, order+2))
        >>> my_quad(2)
        (array([0. , 0.5, 1. ]), array([0.25, 0.25, 0.25]))
        >>> ensure_output(my_quad, order=numpy.array([2]))
        (array([[0. , 0.5, 1. ]]), array([0.25, 0.25, 0.25]))

    """
    kwargs = {key: (value.item() if isinstance(value, numpy.ndarray) else value)
              for key, value in kwargs.items()}
    abscissas, weights = quad_func(**kwargs)
    abscissas = numpy.atleast_2d(abscissas)
    assert abscissas.ndim == 2
    assert weights.ndim == 1
    assert abscissas.shape[-1] == len(weights)
    return abscissas, weights


def univariate_to_multivariate(quad_func, **kwargs):
    """
    Turn a univariate quadrature rule into a multivariate rule.

    The one-dimensional quadrature functions are combined into a multivariate
    through tensor-product. The dimensionality is inferred from the keyword
    arguments. Weights are adjusted to correspond to the multivariate scheme.

    Args:
        quad_func (Callable):
            Function that creates quadrature abscissas and weights on the unit
            interval.
        kwargs (Any):
            Keyword arguments passed to `quad_func`. If numerical value is
            provided, it is used to infer the dimensions of the multivariate
            output. Non-numerical values are passed as is.

    Returns:
        Same as ``quad_func(order, **kwargs)`` except with multivariate
        supported.

    Examples:
        >>> def my_quad(order):
        ...     return (numpy.linspace(0, 1, order+1)[numpy.newaxis],
        ...             1./numpy.full(order+1, order+2))
        >>> my_quad(1)
        (array([[0., 1.]]), array([0.33333333, 0.33333333]))
        >>> my_quad(2)
        (array([[0. , 0.5, 1. ]]), array([0.25, 0.25, 0.25]))
        >>> abscissas, weights = univariate_to_multivariate(
        ...     my_quad, order=numpy.array([1, 2, 1]))
        >>> abscissas
        array([[0. , 0. , 0. , 0. , 0. , 0. , 1. , 1. , 1. , 1. , 1. , 1. ],
               [0. , 0. , 0.5, 0.5, 1. , 1. , 0. , 0. , 0.5, 0.5, 1. , 1. ],
               [0. , 1. , 0. , 1. , 0. , 1. , 0. , 1. , 0. , 1. , 0. , 1. ]])
        >>> weights.round(6)
        array([0.027778, 0.027778, 0.027778, 0.027778, 0.027778, 0.027778,
               0.027778, 0.027778, 0.027778, 0.027778, 0.027778, 0.027778])
        >>> univariate_to_multivariate(my_quad, order=numpy.array([1]))
        (array([[0., 1.]]), array([0.33333333, 0.33333333]))

    """
    sizables = {key: value for key, value in kwargs.items()
                if isinstance(value, (int, float, numpy.ndarray))}
    sizables["_"] = numpy.zeros(len(kwargs.get("domain", [0])))
    nonsizables = {key: value for key, value in kwargs.items()
                   if not isinstance(value, (int, float, numpy.ndarray))}
    keys = list(sizables)
    args = numpy.broadcast_arrays(*[sizables[key] for key in keys])
    assert args[0].ndim == 1
    sizables = {key: value for key, value in zip(keys, args)}
    del sizables["_"]

    results = []
    for idx in range(args[0].size):
        sizable = kwargs.copy()
        sizable.update({key: value[idx].item()
                        for key, value in sizables.items()})
        abscissas, weights = quad_func(**sizable)
        results.append((abscissas.ravel(), weights))

    abscissas, weights = zip(*results)
    return combine_quadrature(abscissas, weights)


def distribution_to_domain(quad_func, distribution, **kwargs):
    """
    Integrate over a distribution domain.

    Adjust weights to account for probability density.

    Args:
        quad_func (Callable):
            Function that creates quadrature abscissas and weights. Must accept
            the arguments ``lower`` and ``upper`` to define the interval it is
            integrating over.
        distribution (:class:`chaospy.Distribution`):
            Distribution to adjust quadrature scheme to.
        kwargs (Any):
            Extra keyword arguments passed to `quad_func`. Can not include the
            arguments ``lower`` and ``upper`` as they are taken from
            ``distribution``.

    Returns:
        Same as ``quad_func(order, **kwargs)`` except arguments ``lower`` and
        ``upper`` are now replaced with a new ``distribution`` argument.

    Examples:
        >>> def my_quad(lower=0, upper=1):
        ...     return (numpy.linspace(lower, upper, 5).reshape(1, -1)[:, 1:-1],
        ...             1./numpy.full(3, 4))
        >>> my_quad()
        (array([[0.25, 0.5 , 0.75]]), array([0.25, 0.25, 0.25]))
        >>> distribution_to_domain(my_quad, chaospy.Uniform(-1, 1))
        (array([[-0.5,  0. ,  0.5]]), array([0.125, 0.125, 0.125]))
        >>> distribution_to_domain(my_quad, chaospy.Beta(2, 2))
        (array([[0.25, 0.5 , 0.75]]), array([0.225, 0.3  , 0.225]))
        >>> distribution_to_domain(my_quad, chaospy.Exponential(1))  # doctest: +NORMALIZE_WHITESPACE
        (array([[ 8.05924772, 16.11849545, 24.17774317]]),
         array([2.32578431e-02, 7.35330570e-06, 2.32485465e-09]))

    """
    assert isinstance(distribution, chaospy.Distribution)
    assert "lower" not in kwargs
    assert "upper" not in kwargs
    lower = distribution.lower
    upper = distribution.upper
    abscissas, weights = quad_func(lower=lower, upper=upper, **kwargs)

    # Sometimes edge samples (inside the distribution domain) falls out again from simple
    # rounding errors. Edge samples needs to be adjusted.
    eps = 1e-14*(distribution.upper-distribution.lower)
    abscissas_ = numpy.clip(abscissas.T, distribution.lower+eps, distribution.upper-eps).T
    weights_ = weights*distribution.pdf(abscissas_).ravel()
    weights = weights_*numpy.sum(weights)/(numpy.sum(weights_)*numpy.prod(upper-lower))
    return abscissas, weights


def split_into_segments(quad_func, order, segments, **kwargs):
    """
    Split a quadrature rule on ta unit interval to multiple segments.

    If both quadrature function includes the abscissas endpoints 0 and 1, then
    the endpoints of each subsequent interval is collapsed and their weights
    added together. This to avoid nodes from being repeated.

    Args:
        quad_func (Callable):
            Function that creates quadrature abscissas and weights on the unit
            interval.
        order (int):
            The quadrature order passed to the quadrature function.
        segments (int, Sequence[float]):
            The number segments to split the interval on. If sequence is
            provided, use as segment edges instead.
        kwargs (Any):
            Extra keyword arguments passed to `quad_func`.

    Returns:
        Same as ``quad_func(order, **kwargs)`` except segmented into
        subintervals.

    Examples:
        >>> def my_quad(order):
        ...     return (numpy.linspace(0, 1, order+1)[numpy.newaxis],
        ...             1./numpy.full(order+1, order+2))
        >>> my_quad(2)
        (array([[0. , 0.5, 1. ]]), array([0.25, 0.25, 0.25]))
        >>> split_into_segments(my_quad, 4, segments=2)  # doctest: +NORMALIZE_WHITESPACE
        (array([[0.  , 0.25, 0.5 , 0.75, 1.  ]]),
         array([0.125, 0.125, 0.25 , 0.125, 0.125]))
        >>> split_into_segments(my_quad, 4, segments=3)  # doctest: +NORMALIZE_WHITESPACE
        (array([[0.        , 0.16666667, 0.33333333, 0.66666667, 1.        ]]),
         array([0.08333333, 0.08333333, 0.19444444, 0.22222222, 0.11111111]))
        >>> split_into_segments(my_quad, 4, segments=[0.2, 0.8])  # doctest: +NORMALIZE_WHITESPACE
        (array([[0. , 0.1, 0.2, 0.5, 0.8, 0.9, 1. ]]),
         array([0.05, 0.05, 0.2 , 0.15, 0.2 , 0.05, 0.05]))

    """
    segments = numpy.array(segments)
    if segments.size == 1:
        segments = int(segments)
        if segments == 1 or order <= 2:
            return quad_func(order=order, **kwargs)
        if not segments:
            segments = int(numpy.sqrt(order))
        assert segments < order, "few samples to distribute than intervals"
        nodes = numpy.linspace(0, 1, segments+1)
    else:
        nodes, segments = numpy.hstack([[0], segments, [1]]), len(segments)

    abscissas = []
    weights = []
    for idx, (lower, upper) in enumerate(zip(nodes[:-1], nodes[1:])):

        assert lower < upper
        order_ = order//segments + (idx < (order%segments))
        abscissa, weight = quad_func(order=order_, **kwargs)
        weight = weight*(upper-lower)
        abscissa = (abscissa.T*(upper-lower)+lower).T
        if abscissas and numpy.allclose(abscissas[-1][:, -1], lower):
            weights[-1][-1] += weight[0]
            abscissa = abscissa[:, 1:]
            weight = weight[1:]
        abscissas.append(abscissa)
        weights.append(weight)

    abscissas = numpy.hstack(abscissas)
    weights = numpy.hstack(weights)
    assert abscissas.shape == (1, len(weights))
    return abscissas, weights


def scale_quadrature(quad_func, order, lower, upper, **kwargs):
    """
    Scale quadrature rule designed for unit interval to an arbitrary interval.

    Args:
        quad_func (Callable):
            Function that creates quadrature abscissas and weights on the unit
            interval.
        order (int):
            The quadrature order passed to the quadrature function.
        lower (float):
            The new lower limit for the quadrature function.
        upper (float):
            The new upper limit for the quadrature function.
        kwargs (Any):
            Extra keyword arguments passed to `quad_func`.

    Returns:
        Same as ``quad_func(order, **kwargs)`` except scaled to a new interval.


    Examples:
        >>> def my_quad(order):
        ...     return (numpy.linspace(0, 1, order+1)[numpy.newaxis],
        ...             1./numpy.full(order+1, order+2))
        >>> my_quad(2)
        (array([[0. , 0.5, 1. ]]), array([0.25, 0.25, 0.25]))
        >>> scale_quadrature(my_quad, 2, lower=0, upper=2)
        (array([[0., 1., 2.]]), array([0.5, 0.5, 0.5]))
        >>> scale_quadrature(my_quad, 2, lower=-0.5, upper=0.5)
        (array([[-0.5,  0. ,  0.5]]), array([0.25, 0.25, 0.25]))

    """
    abscissas, weights = quad_func(order=order, **kwargs)
    assert numpy.all(abscissas >= 0) and numpy.all(abscissas <= 1)
    assert numpy.sum(weights) <= 1+1e-10
    assert numpy.sum(weights > 0)
    weights = weights*(upper-lower)
    abscissas = (abscissas.T*(upper-lower)+lower).T
    return abscissas, weights
