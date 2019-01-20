"""Common functions and attributes used across submodules."""
import inspect


class DependencyError(ValueError):
    """Error that occurs with bad stochastic dependency structures."""


def contains_call_signature(caller, key):
    """
    Check if a function or method call signature contains a specific
    argument.

    Args:
        caller (Callable):
            Method or function to check if signature is contain in.
        key (str):
            Signature to look for.

    Returns:
        True if ``key`` exits in ``caller`` call signature.

    Examples:
        >>> def foo(param): pass
        >>> contains_call_signature(foo, "param")
        True
        >>> contains_call_signature(foo, "not_param")
        False
        >>> class Bar:
        ...     def baz(self, param): pass
        >>> bar = Bar()
        >>> contains_call_signature(bar.baz, "param")
        True
        >>> contains_call_signature(bar.baz, "not_param")
        False
    """
    try:
        args = inspect.signature(caller).parameters
    except AttributeError:
        args = inspect.getargspec(caller).args
    return key in args
