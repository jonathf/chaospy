import inspect

import numpy
import chaospy.quad


def get_function(rule, domain, **parameters):
    """
    Create a quadrature function and set default parameter values.

    Args:
        rule (str) : Name of quadrature rule defined in `QUAD_FUNCTIONS`.
        domain (Dist, array_like) : Defines `lower` and `upper` that is passed
            quadrature rule. If `Dist`, `domain` is renamed to `dist` and also
            passed.
        **parameters (optional) : Redefining of the parameter defaults. Only
            add parameters that the quadrature rule expect.
    Returns:
        (callable) : Function that can be called only using argument `order`.
    """
    if isinstance(domain, chaospy.dist.Dist):
        lower, upper = domain.range()
        parameters["dist"] = domain
    else:
        lower, upper = numpy.array(domain)
    parameters["lower"] = lower
    parameters["upper"] = upper

    quad_function = chaospy.quad.collection.QUAD_FUNCTIONS[rule]
    parameters_spec = inspect.getargspec(quad_function)[0]
    parameters_spec = {key: None for key in parameters_spec}
    del parameters_spec["order"]

    for key in parameters_spec:
        if key in parameters:
            parameters_spec[key] = parameters[key]

    def _quad_function(order, *args, **kws):
        """Implementation of quadrature function."""
        params = parameters_spec.copy()
        params.update(kws)
        return quad_function(order, *args, **params)

    return _quad_function
