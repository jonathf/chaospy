"""Function for handling stochastic dependencies."""
from .common import DependencyError


def sorted_dependencies(dist, reverse=False):
    """
    Extract all underlying dependencies from a distribution sorted
    topologically.

    Uses depth-first algorithm. See more here:

    Args:
        dist (Dist):
            Distribution to extract dependencies from.
        reverse (bool):
            If True, place dependencies in reverse order.

    Returns:
        dependencies (List[Dist]):
            All distribution that ``dist`` is dependent on, sorted
            topologically, including itself.

    Examples:
        >>> dist1 = chaospy.Uniform()
        >>> dist2 = chaospy.Normal(dist1)
        >>> print(sorted_dependencies(dist1))
        [Uniform(lower=0, upper=1), Mul(uniform(), [0.5]), uniform()]
        >>> print(sorted_dependencies(dist2)) # doctest: +NORMALIZE_WHITESPACE
        [Normal(mu=Uniform(lower=0, upper=1), sigma=1),
         Uniform(lower=0, upper=1),
         Mul(uniform(), [0.5]),
         uniform(),
         Mul(normal(), [1.]),
         normal()]
        >>> dist1 in sorted_dependencies(dist2)
        True
        >>> dist2 in sorted_dependencies(dist1)
        False

    Raises:
        DependencyError:
            If the dependency DAG is cyclic, dependency resolution is not
            possible.

    See also:
        Depth-first algorithm section:
        https://en.wikipedia.org/wiki/Topological_sorting
    """
    from .. import baseclass

    collection = [dist]

    # create DAG as list of nodes and edges:
    nodes = [dist]
    edges = []
    pool = [dist]
    while pool:
        dist = pool.pop()
        for key in sorted(dist.prm):
            value = dist.prm[key]
            if not isinstance(value, baseclass.Dist):
                continue
            if (dist, value) not in edges:
                edges.append((dist, value))
            if value not in nodes:
                nodes.append(value)
                pool.append(value)

    # temporary stores used by depth first algorith.
    permanent_marks = set()
    temporary_marks = set()

    def visit(node):
        """Depth-first topological sort algorithm."""
        if node in permanent_marks:
            return
        if node in temporary_marks:
            raise DependencyError("cycles in dependency structure.")

        nodes.remove(node)
        temporary_marks.add(node)

        for node1, node2 in edges:
            if node1 is node:
                visit(node2)

        temporary_marks.remove(node)
        permanent_marks.add(node)
        pool.append(node)

    # kickstart algorithm.
    while nodes:
        node = nodes[0]
        visit(node)

    if not reverse:
        pool = list(reversed(pool))

    return pool


def get_dependencies(*distributions):
    """
    Get underlying dependencies that are shared between distributions.

    If more than two distributions are provided, any pair-wise dependency
    between any two distributions are included, implying that an empty set is
    returned if and only if the distributions are i.i.d.

    Args:
        distributions:
            Distributions to check for dependencies.

    Returns:
        dependencies (List[Dist]):
            Distributions dependency shared at least between at least one pair
            from ``distributions``.

    Examples:
        >>> dist1 = chaospy.Uniform(1, 2)
        >>> dist2 = chaospy.Uniform(1, 2) * dist1
        >>> dist3 = chaospy.Uniform(3, 5)
        >>> print(chaospy.get_dependencies(dist1, dist2))
        [uniform(), Mul(uniform(), [0.5]), Uniform(lower=1, upper=2)]
        >>> print(chaospy.get_dependencies(dist1, dist3))
        []
        >>> print(chaospy.get_dependencies(dist2, dist3))
        []
        >>> print(chaospy.get_dependencies(dist1, dist2, dist3))
        [uniform(), Mul(uniform(), [0.5]), Uniform(lower=1, upper=2)]
    """
    from .. import baseclass
    distributions = [
        sorted_dependencies(dist) for dist in distributions
        if isinstance(dist, baseclass.Dist)
    ]

    dependencies = list()
    for idx, dist1 in enumerate(distributions):
        for dist2 in distributions[idx+1:]:
            dependencies.extend([dist for dist in dist1 if dist in dist2])

    return sorted(dependencies)
