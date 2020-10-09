"""Distribution utility functions."""
import os
import logging
from contextlib import wraps
from itertools import permutations

import numpy
import chaospy


def check_dependencies(distribution):
    """
    Check if the dependency structure is valid.

    Rosenblatt transformations, density calculations etc. assumes that the
    input and output of transformation is the same. It also assumes that there
    is a order defined in `distribution._rotation` so an decomposition on the
    form `p(x0), p(x1|x0), ...` is possible.

    This should be checked on function calls, not init, as it does not apply to
    e.g. moment and three-terms-recurrence calculations.

    Args:
        distribution:
            The distribution to verify is correctly set up.

    Raises:
        StochasticallyDependentError:
            If invalid dependency structure is present.

    Examples:
        >>> dist = chaospy.Uniform(0, 1)
        >>> chaospy.check_dependencies(dist)
        >>> chaospy.check_dependencies(chaospy.Normal(mu=dist))
        Traceback (most recent call last):
            ...
        chaospy.StochasticallyDependentError: \
Normal(mu=Uniform(), sigma=1) has dangling dependencies
    """
    current = set()
    for idx in distribution._rotation:
        length = len(current)
        current.update(distribution._dependencies[idx])
        if len(current) != length+1:
            raise chaospy.StochasticallyDependentError(
                "%s has dangling dependencies" % distribution)


_EXCEPTION_IN_PROGRESS = False


def report_on_exception(method):
    """
    Method decorators for getting more verbose output.

    Will output the function name and call signature to logger when an
    exception is raised. But only during testing or if the environment variable
    `CHAOSPY_DEBUG=1` is set.

    Args:
        method:
            Method to be wrapped.

    Returns:
        Same as 'method', but wrapped to include exception logger.

    """
    logger = logging.getLogger(__name__)

    @wraps(method)
    def wrapper_method(self, *args, **kwargs):
        """Method to wrap 'method'."""
        global _EXCEPTION_IN_PROGRESS
        _EXCEPTION_IN_PROGRESS = False
        try:
            ret_val = method(self, *args, **kwargs)
        except Exception as err:
            if not _EXCEPTION_IN_PROGRESS:
                _EXCEPTION_IN_PROGRESS = True
                args = ",\n    ".join([repr(arg) for arg in args]+
                                    ["%s=%r" % (key, val) for key, val in kwargs.items()])
                logger.warning("failure:\n%s.%s(\n    %s\n)", self, method.__name__, args)
            else:
                logger.warning("failure: %s.%s", self, method.__name__)
            raise
        return ret_val

    if os.environ.get("CHAOSPY_DEBUG", "") == "1":
        method = wrapper_method
    return method


def shares_dependencies(*distributions):
    """
    Check if a collection of distributions shares dependencies.

    Internal dependencies are ignored here. It is only intra-dependencies that
    are checked. Anything else than a distribution is ignored.

    Args:
        distributions (Sequence[chaospy.Distribution]):
            The distributions to check shares dependencies or not.

    Returns:
        True if there are dependencies between the provided distributions.

    Examples:
        >>> dist = chaospy.Normal()
        >>> chaospy.shares_dependencies(dist, 4)
        False
        >>> chaospy.shares_dependencies(dist, chaospy.Normal())
        False
        >>> chaospy.shares_dependencies(dist, chaospy.Normal(dist))
        True

    """
    distributions = [dist for dist in distributions
                     if isinstance(dist, chaospy.Distribution)]
    if len(distributions) == 1:
        return False
    dependencies = [{dep for deps in dist._dependencies for dep in deps}
                    for dist in distributions]
    for deps1, deps2 in permutations(dependencies, 2):
        if deps1.intersection(deps2):
            return True
    return False


def declare_dependencies(
        distribution,
        parameters,
        rotation=None,
        is_operator=False,
        dependency_type="iid",
        length=None,
):
    """
    Convenience function for declaring distribution dependencies.

    Iterates through parameters to figure out what a distribution dependency
    structure should be.

    Args:
        distribution (chaospy.Distribution):
            The distributions to to declare dependencies for.
        parameters (Dict[str, Any]):
            The distribution parameters that should be included in the
            declaration.
        is_operator (bool):
            Operators do not themselves contain uncertainty, but only inherits
            from parameters and/or wrapped distribution.
        wrapper_dist (Optional[chaospy.Distribution]):
            Distributions that are thin-wrappers to some other distribution
            should inherent dependencies.
    """
    parameters = parameters.copy()
    for name, parameter in list(parameters.items()):
        if not isinstance(parameter, chaospy.Distribution):
            parameters[name] = numpy.atleast_1d(parameter)
    if length is None:
        if rotation is None:
            length = max([len(parameter) for parameter in parameters.values()]+[1])
        else:
            length = len(rotation)
    if rotation is None:
        rotation = numpy.arange(length, dtype=int)
    else:
        rotation = numpy.asarray(rotation)

    if is_operator:
        dependencies = [set() for _ in range(length)]
    else:
        dependencies = init_dependencies(distribution, rotation, dependency_type=dependency_type)

    for name, parameter in list(parameters.items()):
        if isinstance(parameter, chaospy.Distribution):
            if len(parameter) != length:
                raise chaospy.StochasticallyDependentError(
                    "dependencies must be same length as parent")
            for dep1, dep2 in zip(dependencies, parameter._dependencies):
                dep1.update(dep2)
        else:
            parameters[name] = parameter*numpy.ones(length, dtype=int)
    return dependencies, parameters, rotation


DISTRIBUTION_IDENTIFIERS = {}

def init_dependencies(
        distribution,
        rotation,
        dependency_type="iid",
):
    """
    Declare stochastic dependency to an underlying random variable.

    Args:
        distribution (chaospy.Distribution):
            Distribution to declare dependency for.
        count (int):
            The number of variables to declare.

    Returns:
        (List[Set[int]]):
            Unique integer identifiers that represents dependencies.

    """
    rotation = numpy.asarray(rotation)
    assert rotation.dtype == int and rotation.ndim == 1
    length = len(rotation)
    next_new_id = len(DISTRIBUTION_IDENTIFIERS)
    new_identifiers = numpy.arange(next_new_id, next_new_id+length, dtype=int)
    for idx in new_identifiers:
        DISTRIBUTION_IDENTIFIERS[idx] = distribution

    if dependency_type == "iid":
        dependencies = [{idx} for idx in new_identifiers[rotation]]

    elif dependency_type == "accumulate":
        accumulant = set()
        dependencies = [None]*length
        for idx in rotation:
            accumulant.add(new_identifiers[idx])
            dependencies[idx] = accumulant.copy()

    return dependencies


def format_repr_kwargs(**parameters):
    out = []

    defaults_only = True
    for name, (param, default) in list(parameters.items()):
        defaults_only &= (
            isinstance(param, (int, float)) and param == default)
        if isinstance(param, numpy.ndarray):
            parameters[name] = (param.tolist(), default)
    if defaults_only:
        return []
    return ["%s=%s" % (key, parameters[key][0]) for key in sorted(parameters)]
