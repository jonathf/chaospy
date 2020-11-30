"""Distribution with user-provided methods."""
import numpy
import chaospy

from .simple import SimpleDistribution


class UserDistribution(SimpleDistribution):
    """
    Distribution with user-provided methods.

    The internals of this distribution is provided in the constructor.

    Examples:
        >>> def cdf(x_loc, lo, up):
        ...     '''Cumulative distribution function.'''
        ...     return (x_loc-lo)/(up-lo)
        >>> def pdf(x_loc, lo, up):
        ...     '''Probability density function.'''
        ...     return 1./(up-lo)
        >>> def lower(lo, up):
        ...     '''Lower bounds function.'''
        ...     return lo
        >>> def upper(lo, up):
        ...     '''Upper bounds function.'''
        ...     return up
        >>> distribution = chaospy.UserDistribution(
        ...     cdf, pdf, lower, upper, parameters=dict(lo=-1, up=1))
        >>> distribution
        UserDistribution(<function ..., parameters=dict(lo=-1, up=1))
        >>> distribution.fwd(numpy.linspace(-2, 2, 7)).round(4)
        array([0.    , 0.    , 0.1667, 0.5   , 0.8333, 1.    , 1.    ])
        >>> distribution.pdf(numpy.linspace(-2, 2, 7)).round(4)
        array([0. , 0. , 0.5, 0.5, 0.5, 0. , 0. ])
        >>> distribution.inv(numpy.linspace(0, 1, 7)).round(4)
        array([-1.    , -0.6667, -0.3333,  0.    ,  0.3333,  0.6667,  1.    ])
        >>> distribution.lower, distribution.upper
        (array([-1.]), array([1.]))

    """

    _cdf = None  # required by class constructor

    def __init__(
        self,
        cdf,
        pdf=None,
        lower=None,
        upper=None,
        ppf=None,
        mom=None,
        ttr=None,
        parameters=None
    ):
        """
        Args:
            cdf (Callable[[numpy.ndarray, ...], numpy.ndarray]):
                Cumulative distribution function.
            pdf (Callable[[numpy.ndarray, ...], numpy.ndarray]):
                Probability density function.
            lower (Callable[[...], numpy.ndarray]):
                Lower boundary.
            upper (Callable[[...], numpy.ndarray]):
                Upper boundary.
            ppf (Callable[[numpy.ndarray, ...], numpy.ndarray]):
                Point percentile function.
            mom (Callable[[numpy.ndarray, ...], numpy.ndarray]):
                Raw moment generator.
            ttr (Callable[[numpy.ndarray, ...], numpy.ndarray]):
                Three terms recurrence coefficient generator.
            parameters (Dict[str, numpy.ndarray]):
                Parameters to pass to each of the distribution methods.

        """
        self._cdf = cdf
        repr_args = [str(cdf)]
        if ppf is None and (lower is None or upper is None):
            raise chaospy.UnsupportedFeature("either ppf or lower+upper should be provided.")
        if pdf is not None:
            repr_args.append("pdf=%s" % pdf)
            self._pdf = pdf
        if lower is not None:
            repr_args.append("lower=%s" % lower)
            self._lower = lower
        if upper is not None:
            repr_args.append("upper=%s" % upper)
            self._upper = upper
        if ppf is not None:
            repr_args.append("ppf=%s" % ppf)
            self._ppf = ppf
        if mom is not None:
            repr_args.append("mom=%s" % mom)
            self._mom = mom
        if ttr is not None:
            repr_args.append("ttr=%s" % ttr)
            self._ttr = ttr

        parameters = parameters if parameters else {}
        if parameters:
            params = [("%s=%s" % (key, parameters[key]))
                        for key in sorted(parameters)]
            repr_args.append("parameters=dict(%s)" % ", ".join(params))

        super(UserDistribution, self).__init__(
            repr_args=repr_args, parameters=parameters)

    def _lower(self, **parameters):
        x_loc = 0.
        for param in parameters.values():
            x_loc, _ = numpy.broadcast_arrays(x_loc, param)
        return self._ppf(x_loc, **parameters)

    def _upper(self, **parameters):
        x_loc = 1.
        for param in parameters.values():
            x_loc, _ = numpy.broadcast_arrays(x_loc, param)
        return self._ppf(x_loc, **parameters)
