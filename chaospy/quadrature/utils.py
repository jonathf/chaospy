from __future__ import division
from functools import partial
import numpy
import chaospy


def ensure_output(quad_func, **kwargs):
    kwargs = {key: (value.item() if isinstance(value, numpy.ndarray) else value)
              for key, value in kwargs.items()}
    abscissas, weights = quad_func(**kwargs)
    abscissas = numpy.atleast_2d(abscissas)
    assert abscissas.ndim == 2
    assert weights.ndim == 1
    assert abscissas.shape[-1] == len(weights)
    return abscissas, weights


def ensure_input(quad_func, **kwargs):
    sizables = {key: value for key, value in kwargs.items()
                if isinstance(value, (int, float, numpy.ndarray))}
    nonsizables = {key: value for key, value in kwargs.items()
                   if not isinstance(value, (int, float, numpy.ndarray))}

    sizables["_"] = numpy.zeros(len(kwargs.get("domain", [0])))
    keys = list(sizables)
    args = numpy.broadcast_arrays(*[sizables[key] for key in keys])
    assert args and args[0].ndim <= 1, kwargs
    sizables = {key: value for key, value in zip(keys, args)}
    del sizables["_"]
    return quad_func(**sizables, **nonsizables)



def distribution_to_domain(quad_func, distribution, **kwargs):
    assert isinstance(distribution, chaospy.Distribution)
    abscissas, weights = quad_func(
        lower=distribution.lower,
        upper=distribution.upper,
        **kwargs
    )
    # Sometimes edge samples (inside the distribution domain) falls out again from simple
    # rounding errors. Edge samples needs to be adjusted.
    eps = 1e-14*(distribution.upper-distribution.lower)
    abscissas_ = numpy.clip(abscissas.T, distribution.lower+eps, distribution.upper-eps).T
    weights = weights*distribution.pdf(abscissas_).ravel()
    weights /= numpy.sum(weights)
    return abscissas, weights


def univariate_to_multivariate(quad_func, **kwargs):
    sizables = {key: value for key, value in kwargs.items()
                if isinstance(value, (int, float, numpy.ndarray))}
    keys = list(sizables)
    args = numpy.broadcast_arrays(*[sizables[key] for key in keys])
    if not args[0].ndim:
        return quad_func(**kwargs)

    assert args[0].ndim == 1
    sizables = {key: value for key, value in zip(keys, args)}

    results = []
    for idx in range(args[0].size):
        sizable = kwargs.copy()
        sizable.update({key: value[idx].item() for key, value in sizables.items()})
        abscissas, weights = quad_func(**sizable)
        results.append((abscissas.ravel(), weights))

    abscissas, weights = zip(*results)
    return combine_quadrature(abscissas, weights)


def split_into_segments(quad_func, order, segments, **kwargs):
    if segments == 1 or order <= 2:
        return quad_func(order=order, **kwargs)
    if not segments:
        segments = int(numpy.sqrt(order))
    assert segments < order, "few samples to distribute than intervals"
    abscissas = []
    weights = []

    nodes = numpy.linspace(0, 1, segments+1)
    for idx, (lower, upper) in enumerate(zip(nodes[:-1], nodes[1:])):

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


def scale_samples(quad_func, order, lower, upper, **kwargs):
    abscissas, weights = quad_func(order=order, **kwargs)
    weights = weights*(upper-lower)
    abscissas = (abscissas.T*(upper-lower)+lower).T
    return abscissas, weights


def combine(args):
    """
    All linear combination of a list of list.

    Args:
        args (numpy.ndarray):
            List of input arrays.  Components to take linear combination of
            with ``args[i].shape == (N[i], M[i])`` where ``N`` is to be taken
            linear combination of and ``M`` is constant.  ``M[i]`` is set to
            1 if missing.

    Returns:
        (numpy.array):
            Matrix of combinations with
            ``shape == (numpy.prod(N), numpy.sum(M))``.

    Examples:
        >>> A, B = [1,2], [[4,4],[5,6]]
        >>> combine([A, B])
        array([[1, 4, 4],
               [1, 5, 6],
               [2, 4, 4],
               [2, 5, 6]])
    """
    args = [numpy.asarray(arg).reshape(len(arg), -1) for arg in args]
    shapes = [arg.shape for arg in args]

    size = numpy.prod(shapes, 0)[0]*numpy.sum(shapes, 0)[1]

    out = args[0]
    for arg in args[1:]:
        out = numpy.hstack([
            numpy.tile(out, len(arg)).reshape(-1, out.shape[1]),
            numpy.tile(arg.T, len(out)).reshape(arg.shape[1], -1).T,
        ])
    return out


def combine_quadrature(
        abscissas,
        weights,
        domain=(),
):
    """
    Create all linear combinations of all abscissas and weights. If ``domain``
    is provided, also scale from assumed (0, 1) to said domain.

    Args:
        abscissas (List[numpy.ndarray]):
            List of abscissas to be combined.
        weights (List[numpy.ndarray]):
            List of weights to be combined.
        domain (Optional[Tuple[numpy.ndarray, numpy.ndarray]]):
            Domain to scale to.

    Returns:
        (Tuple[numpy.ndarray, numpy.ndarray]):
            Same as ``abscissas`` and ``weights``, but combined and flatten
            such that ``abscissas.shape == (dim, len(weights))``.
    """
    dim = len(abscissas)
    abscissas = combine(abscissas)
    weights = combine(weights)

    if domain:
        abscissas = (domain[1]-domain[0])*abscissas + domain[0]
        weights = weights*(domain[1]-domain[0])

    abscissas = abscissas.T.reshape(dim, -1)
    weights = numpy.prod(weights, -1)

    assert len(weights.shape) == 1
    assert abscissas.shape == (dim,) + weights.shape, (abscissas, weights)

    return abscissas, weights
