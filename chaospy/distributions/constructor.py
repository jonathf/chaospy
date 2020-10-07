"""
Custom distribution constructor.

Example usage
-------------

Construct distribution from scratch::

    >>> MyUniform = chaospy.construct(
    ...     cdf=lambda self, x, lo, up: (x-lo)/(up-lo),
    ...     lower=lambda self, lo, up: lo,
    ...     upper=lambda self, lo, up: up,
    ... )

Evaluate distribution::

    >>> uniform = MyUniform(lo=-1, up=1)
    >>> uniform.pdf(numpy.linspace(-2, 2, 12))
    array([0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. ])
"""
import types

import chaospy
from .baseclass import SimpleDistribution

LEGAL_ATTRS = {
    "cdf": "_cdf", "lower": "_lower", "upper": "_upper",
    "pdf": "_pdf", "ppf": "_ppf", "mom": "_mom",
    "ttr": "_ttr", "fwd_cache": "_fwd_cache", "inv_cache": "_inv_cache",
    "doc": "__doc__",
    "str": "_str"
}

class ConstructedDistribution(SimpleDistribution):

    def __init__(self, **kwargs):
        super(ConstructedDistribution, self).__init__(
            parameters=kwargs)

    def _cdf(self, xloc, **kwargs):
        pass

def construct(parent=None, defaults=None, **kwargs):
    """
    Random variable constructor.

    Args:
        cdf:
            Cumulative distribution function. Optional if ``parent`` is used.
        lower:
            Lower boundary. Optional if ``parent`` or ``ppf`` is present.
        upper:
            Upper boundary. Optional if ``parent`` or ``ppf`` is present.
        parent (SimpleDistribution):
            Distribution used as basis for new distribution. Any other argument
            that is omitted will instead take is function from ``parent``.
        doc (str):
            Documentation for the distribution.
        str (str, :py:data:typing.Callable):
            Pretty print of the variable.
        pdf:
            Probability density function.
        ppf:
            Point percentile function.
        mom:
            Raw moment generator.
        ttr:
            Three terms recurrence coefficient generator.
        init:
            Custom initialiser method.
        defaults (dict):
            Default values to provide to initialiser.

    Returns:
        (SimpleDistribution):
            New custom distribution.
    """

    for key in kwargs:
        assert key in LEGAL_ATTRS, "{} is not legal input".format(key)
    if parent is not None:
        for key, value in LEGAL_ATTRS.items():
            if key not in kwargs and hasattr(parent, value):
                    kwargs[key] = getattr(parent, value)

    assert "cdf" in kwargs, "cdf function must be defined"
    if "ppf" not in kwargs:
        assert "lower" in kwargs, "lower function must be defined"
        assert "upper" in kwargs, "upper function must be defined"
    if "str" in kwargs and isinstance(kwargs["str"], str):
        string = kwargs.pop("str")
        kwargs["str"] = lambda *args, **kwargs: string
    defaults = defaults if defaults else {}
    for key in defaults:
        assert key in LEGAL_ATTRS, "invalid default value {}".format(key)


    def custom_distribution(**kws):

        prm = defaults.copy()
        prm.update(kws)
        dist = ConstructedDistribution(**prm)

        for key, function in kwargs.items():
            attr_name = LEGAL_ATTRS[key]
            setattr(dist, attr_name, types.MethodType(function, dist))
        return chaospy.J(dist)

    if "doc" in kwargs:
        custom_distribution.__doc__ = kwargs["doc"]

    return custom_distribution
