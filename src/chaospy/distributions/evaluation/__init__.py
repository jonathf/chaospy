import logging
import inspect

import networkx
import numpy

from ... import quad

from .common import DependencyError
from .parameters import load_parameters as load_inputs

from .density import evaluate_density
from .forward import evaluate_forward
from .inverse import evaluate_inverse
from .bound import evaluate_bound
from .moment import evaluate_moment
from .recurrence_coefficients import evaluate_recurrence_coefficients




def sorted_dependencies(dist, reverse=False):
    from .. import baseclass
    graph = networkx.DiGraph()
    graph.add_node(dist)
    dist_collection = [dist]
    while dist_collection:

        cur_dist = dist_collection.pop()

        values = (value for value in cur_dist.prm.values()
                  if isinstance(value, baseclass.Dist))
        for value in values:
            if value not in graph.node:
                graph.add_node(value)
                dist_collection.append(value)
            graph.add_edge(cur_dist, value)

    if not networkx.is_directed_acyclic_graph(graph):
        raise DependencyError("cycles in dependency structure.")

    sorting = list(networkx.topological_sort(graph))
    if reverse:
        sorting = reversed(sorting)

    for dist in sorting:
        yield dist


def get_dependencies(*distributions):
    from .. import baseclass
    distributions = [
        set(sorted_dependencies(dist)) for dist in distributions
        if isinstance(dist, baseclass.Dist)
    ]
    if len(distributions) <= 1:
        return set()
    intersections = set.intersection(*distributions)
    return intersections




def get_forward_cache(
        distribution,
        cache,
):
    from .. import baseclass
    if not isinstance(distribution, baseclass.Dist):
        return distribution
    if distribution in cache:
        return cache[distribution]
    if hasattr(distribution, "_fwd_cache"):
        return distribution._fwd_cache(cache)
    return distribution

def get_inverse_cache(
        distribution,
        cache,
):
    from .. import baseclass
    if not isinstance(distribution, baseclass.Dist):
        return distribution
    if distribution in cache:
        return cache[distribution]
    if hasattr(distribution, "_inv_cache"):
        return distribution._inv_cache(cache)
    return distribution












