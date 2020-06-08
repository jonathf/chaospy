"""Numpoly-Chaospy compatibility layer."""
from functools import wraps
import numpoly


def compatibility_layer(function):
    """
    Function decorator addressing incompatibility between numpoly and chaospy.

    The issues includes:
    * Enforce that all indeterminant names are on the format ``q<N>``.

    Args:
        function (Callable):
            Function to create wrapper around.

    Returns:
        (Callable):
            Same as `function`, but wrapped to ensure that it is chaospy
            compatible.

    Examples:
        >>> numpoly.monomial(3)
        polynomial([1, q, q**2])
        >>> my_monomial = compatibility_layer(numpoly.monomial)
        >>> my_monomial(3)
        polynomial([1, q0, q0**2])

    """

    @wraps(function)
    def wrapper_func(*args, **kwargs):
        """Wrapper function."""
        with numpoly.global_options(
                default_varname="q",
                force_number_suffix=True,
                varname_filter=r"q\d+",
        ):
            return function(*args, **kwargs)

    return wrapper_func


def wrap_numpoly(namespace):
    """
    Wrap every numpoly functions with `compatibility_layer` decorator.

    This to ensure that the functionality provided in numpoly doesn't cause
    trouble because they are not supported in chaospy.

    Args:
        namespace (Dict[str, Any]):
            Name space where the functions exists. Typically either `locals()`
            or `globals()`. Note that the name space will be changed in-place.

    """
    for name, item in namespace.items():

        # must be a function
        if not callable(item):
            continue

        # must be a numpoly object
        if not item.__module__.startswith("numpoly."):
            continue

        # whitelist
        if name in ("ndpoly",):
            continue

        namespace[name] = compatibility_layer(item)
