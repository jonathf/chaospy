"""
Custom distribution constructor.
"""
import types
from .baseclass import Dist

LEGAL_ATTRS = {
    "cdf": "_cdf", "bnd": "_bnd",
    "pdf": "_pdf", "ppf": "_ppf", "mom": "_mom",
    "ttr": "_ttr", "val": "_val", "doc": "__doc__",
    "str": "_str"
}

def construct(parent=None, defaults=None, advance=False, length=1, **kwargs):
    """
    Random variable constructor.

    Args:
        cdf (callable):
            Cumulative distribution function. Optional if ``parent`` is used.
        bnd (callable):
            Boundary interval. Optional if ``parent`` is used.
        parent (Dist):
            Distribution used as basis for new distribution. Any other argument
            that is omitted will instead take is function from ``parent``.
        doc (str, optional):
            Documentation for the distribution.
        str (str, callable, optional):
            Pretty print of the variable.
        pdf (callable, optional):
            Probability density function.
        ppf (callable, optional):
            Point percentile function.
        mom (callable, optional):
            Raw moment generator.
        ttr (callable, optional):
            Three terms recursion coefficient generator
        val (callable, optional):
            Value function for transferable distributions.
        dep (callable, optional):
            Dependency structure.
        advance (bool):
            If True, advance mode is used. See dist.graph for details.
        length (int):
            If constructing an multivariate random variable, this sets the
            assumed length. Defaults to 1.
        init (callable, optional):
            Custom constructor method.

    Returns:
        dist (Dist) : New custom distribution.
    """
    for key in kwargs:
        assert key in LEGAL_ATTRS, "{} is not legal input".format(key)

    if parent is not None:
        for key, value in LEGAL_ATTRS.items():
            if key not in kwargs and hasattr(parent, value):
                    kwargs[key] = getattr(parent, value)

    assert "cdf" in kwargs, "cdf function must be defined"
    assert "bnd" in kwargs, "bnd function must be defined"
    if "str" in kwargs and isinstance(kwargs["str"], str):
        string = kwargs.pop("str")
        kwargs["str"] = lambda *args, **kwargs: string

    defaults = defaults if defaults else {}
    for key in defaults:
        assert key in LEGAL_ATTRS, "invalid default value {}".format(key)
    defaults["_length"] = length
    defaults["_advance"] = advance

    def custom_distribution(**kws):

        prm = defaults.copy()
        prm.update(kws)
        dist = Dist(**prm)

        for key, function in kwargs.items():
            attr_name = LEGAL_ATTRS[key]
            setattr(dist, attr_name, types.MethodType(function, dist))
        return dist

    if "doc" in kwargs:
        custom_distribution.__doc__ = kwargs["doc"]

    return custom_distribution
