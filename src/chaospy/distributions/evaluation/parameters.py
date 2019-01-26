"""
Parameter loading.

Example usage
-------------

Define a simple distribution::

    >>> class MyDist(chaospy.Dist):
    ...     def _pdf(self, x_data, param1): pass
    >>> dist = MyDist(param1=45)

Normal usage::

    >>> print(load_parameters(dist, "_pdf"))
    {'param1': 45}
    >>> print(load_parameters(dist, "_pdf", parameters={"param1": 35}))
    {'param1': 35}

Define distribution dependencies::

    >>> sub_dist = chaospy.Uniform(0, 1)
    >>> dist = MyDist(param1=sub_dist)

Unresolved usage::

    >>> load_parameters(dist, "_pdf")
    Traceback (most recent call last):
        ...
    chaospy.distributions.baseclass.StochasticallyDependentError: \
evaluating under-defined distribution MyDist(param1=Uniform(lower=0, upper=1)).

Loading combined with caching::

    >>> print(load_parameters(dist, "_pdf", cache={sub_dist: 15}))
    {'param1': 15}

Advanced distributions which has the ``cache`` as a call argument, receive the
cache as the value::

    >>> class MyAdvancedDist(chaospy.Dist):
    ...     def _pdf(self, x_data, param1, cache): pass
    >>> dist = MyAdvancedDist(param1=sub_dist)
    >>> print(load_parameters(dist, "_pdf", cache={sub_dist: 15}))
    {'param1': Uniform(lower=0, upper=1), 'cache': {Uniform(lower=0, upper=1): 15}}
"""
import inspect

from .common import contains_call_signature


def load_parameters(
        distribution,
        method_name,
        parameters=None,
        cache=None,
        cache_key=lambda x:x,
):
    """
    Load parameter values by filling them in from cache.

    Args:
        distribution (Dist):
            The distribution to load parameters from.
        method_name (str):
            Name of the method for where the parameters should be used.
            Typically ``"_pdf"``, ``_cdf`` or the like.
        parameters (optional, Dict[str, Any]):
            Default parameters to use if there are no cache to retrieve. Use
            the distributions internal parameters, if not provided.
        cache (optional, Dict[Dist, numpy.ndarray]):
            A dictionary containing previous evaluations from the stack. If
            a parameters contains a distribution that contains in the cache, it
            will be replaced with the cache value. If omitted, a new one will
            be created.
        cache_key (optional, Callable[Dist, Any])
            Redefine the keys of the cache to suite other purposes.

    Returns:
        Same as ``parameters``, if provided. The ``distribution`` parameter if
        not. In either case, parameters may be updated with cache values (if
        provided) or by ``cache`` if the call signature of ``method_name`` (on
        ``distribution``) contains an ``cache`` argument.
    """
    from .. import baseclass
    if cache is None:
        cache = {}
    if parameters is None:
        parameters = {}
    parameters = {**distribution.prm, **parameters}

    # self aware and should handle things itself:
    if contains_call_signature(getattr(distribution, method_name), "cache"):
        parameters["cache"] = cache

    # dumb distribution and just wants to evaluate:
    else:
        for key, value in parameters.items():
            if isinstance(value, baseclass.Dist):
                value = cache_key(value)
                if value in cache:
                    parameters[key] = cache[value]
                else:
                    raise baseclass.StochasticallyDependentError(
                        "evaluating under-defined distribution {}.".format(distribution))

    return parameters