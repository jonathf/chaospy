"""Distribution utility functions."""
import sys
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
        extra_parameters=None,
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
        rotation (Optional[Sequence[int]]):
            The order of which the dependencies should be resolved.
            Automatically calculated if omitted.
        is_operator (bool):
            Operators do not themselves contain uncertainty, but only inherits
            from parameters and/or wrapped distribution.
        wrapper_dist (Optional[chaospy.Distribution]):
            Distributions that are thin-wrappers to some other distribution
            should inherent dependencies.
        extra_parameters(Optional[Dict[str, Any]]):
            Extra parameters that should be included in the declaration, but
            not considered a direct part of the distribution. Assumed to be
            pre-processed.

    Returns:
        dependencies (List[Set[int]]):
            Dependency reference numbers, one collection per dimension. Single
            element implies stochstically independent.
        parameters (Dict[str, Any]):
            Same as `parameters`, but updated to all conform to the same size.
        rotation (List[int]):
            Same as `rotation` if provided. If not, the automatically
            calculated one is returned.

    """
    extra_parameters = extra_parameters.copy() if extra_parameters else {}
    parameters = parameters.copy()
    for name, parameter in list(parameters.items()):
        if not isinstance(parameter, chaospy.Distribution):
            parameters[name] = numpy.atleast_1d(parameter)
    if length is None:
        if rotation is None:
            length = max([len(parameter) for parameter in extra_parameters.values()]+
                         [len(parameter) for parameter in parameters.values()]+[1])
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
        else:
            parameters[name] = parameter*numpy.ones(length, dtype=int)
    for name, parameter in list(parameters.items())+list(extra_parameters.items()):
        if isinstance(parameter, chaospy.Distribution):
            for dep1, dep2 in zip(dependencies, parameter._dependencies):
                dep1.update(dep2)
    assert len(dependencies) == length
    return dependencies, parameters, rotation


DISTRIBUTION_IDENTIFIERS = {}

def init_dependencies(
        distribution,
        rotation,
        dependency_type="iid",
):
    """
    Declare stochastic dependency to an underlying random variable.

    To be used inside distribution initializers ``__init__`` to map out the
    dependency structure.

    Args:
        distribution (chaospy.Distribution):
            Distribution to declare dependency for.
        rotation (Sequence[int]):
            The order of which the dependencies should be resolved.
        dependency_type (str):
            The type of dependency structure to create. Choose from "iid"
            (independent) and "accumulate" (saturated dependencies).

    Returns:
        (List[Set[int]]):
            Unique integer identifiers that represents dependencies.

    Examples:
        >>> from chaospy.distributions.baseclass.utils import DISTRIBUTION_IDENTIFIERS
        >>> DISTRIBUTION_IDENTIFIERS.clear()
        >>> core = chaospy.Normal()
        >>> core._dependencies
        [{0}]
        >>> DISTRIBUTION_IDENTIFIERS
        {0: normal()}
        >>> distribution = chaospy.Iid(core, 2)
        >>> distribution._dependencies
        [{1}, {2}]
        >>> chaospy.init_dependencies(distribution, [0, 1])
        [{3}, {4}]
        >>> chaospy.init_dependencies(distribution, [0, 1], "accumulate")
        [{5}, {5, 6}]
        >>> chaospy.init_dependencies(distribution, [1, 0])
        [{8}, {7}]
        >>> DISTRIBUTION_IDENTIFIERS[8]
        Iid(Normal(mu=0, sigma=1), 2)

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
    """
    Format arguments for REPR output.

    Remove arguments if value are the same as their defaults. But only if that
    applies to all arguments.

    Args:
        parameters (Tuple[Any, numpy.number]):
            Parameters to format. First value is the parameters and the second
            is the parameters default. Using None for the default means no
            default.

    Returns:
        (List[str]):
            Positional argument to be used for REPR output.

    Examples:
        >>> chaospy.format_repr_kwargs(a=(4, 3), b=(chaospy.Uniform(), 4))
        ['a=4', 'b=Uniform()']
        >>> chaospy.format_repr_kwargs(a=(4, 3), b=(4, 4))
        ['a=4', 'b=4']
        >>> chaospy.format_repr_kwargs(a=(3, 3), b=(4, 4))
        []

    """
    out = []

    defaults_only = True
    for name, (param, default) in list(parameters.items()):
        defaults_only &= (
            (isinstance(param, (int, float)) and param == default) or
            (param is default is None)
        )
        if isinstance(param, numpy.ndarray):
            parameters[name] = (param.tolist(), default)
    if defaults_only:
        return []
    return ["%s=%s" % (key, parameters[key][0]) for key in sorted(parameters)]
