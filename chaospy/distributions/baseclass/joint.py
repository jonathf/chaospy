"""
There are three ways to create a multivariate probability distribution in
``chaospy``: Using the joint constructor
:class:`~chaospy.distributions.operators.joint.J`, the identical independent
distribution constructor: :class:`~chaospy.distributions.baseclass.iid.Iid`,
and to one of the pre-constructed multivariate distribution defined in
:ref:`listdistributions`.

Constructing a multivariate probability distribution can be done using the
:func:`~chaospy.distributions.baseclass.joint.J` constructor. E.g.::

    >>> distribution = chaospy.J(
    ...     chaospy.Normal(0, 1), chaospy.Uniform(0, 1))

The created multivariate distribution behaves much like the univariate case::

    >>> mesh = numpy.mgrid[0.25:0.75:3j, 0.25:0.75:3j]
    >>> mesh
    array([[[0.25, 0.25, 0.25],
            [0.5 , 0.5 , 0.5 ],
            [0.75, 0.75, 0.75]],
    <BLANKLINE>
           [[0.25, 0.5 , 0.75],
            [0.25, 0.5 , 0.75],
            [0.25, 0.5 , 0.75]]])
    >>> distribution.cdf(mesh).round(4)
    array([[0.1497, 0.2994, 0.449 ],
           [0.1729, 0.3457, 0.5186],
           [0.1933, 0.3867, 0.58  ]])
    >>> distribution.pdf(mesh).round(4)
    array([[0.3867, 0.3867, 0.3867],
           [0.3521, 0.3521, 0.3521],
           [0.3011, 0.3011, 0.3011]])
    >>> distribution.sample(6, rule="halton").round(4)
    array([[-1.1503,  0.3186, -0.3186,  1.1503, -1.5341,  0.1573],
           [ 0.4444,  0.7778,  0.2222,  0.5556,  0.8889,  0.037 ]])
    >>> distribution.mom([[2, 4, 6], [1, 2, 3]]).round(10)
    array([0.5 , 1.  , 3.75])

"""
import numpy
import chaospy

from .distribution import Distribution


def sorted_dependencies(dist, cache=None, reverse=False):
    """
    Extract all underlying dependencies from a distribution sorted
    topologically.

    Uses depth-first algorithm. See more here:

    Args:
        dist (Distribution):
            Distribution to extract dependencies from.
        cache (Optional[Dict[Distribution, numpy.ndarray]]):
            Values already evaluated filling out any conditionals
            to make calculations possible.
        reverse (bool):
            If True, place dependencies in reverse order.

    Returns:
        dependencies (List[Distribution]):
            All distribution that ``dist`` is dependent on, sorted
            topologically, including itself.

    Examples:
        >>> dist1 = chaospy.Uniform()
        >>> dist2 = chaospy.Normal(dist1)
        >>> sorted_dependencies(dist1)
        [Uniform()]
        >>> print(sorted_dependencies(dist2)) # doctest: +NORMALIZE_WHITESPACE
        [Normal(mu=Uniform(), sigma=1), Uniform()]
        >>> dist1 in sorted_dependencies(dist2)
        True
        >>> dist2 in sorted_dependencies(dist1)
        False

    Raises:
        StochasticallyDependentError:
            If the dependency DAG is cyclic, dependency resolution is not
            possible.

    See also:
        Depth-first algorithm section:
        https://en.wikipedia.org/wiki/Topological_sorting
    """
    if cache is None:
        cache = {}
    collection = [dist]

    # create DAG as list of nodes and edges:
    nodes = [dist]
    edges = []
    pool = [dist]
    while pool:
        dist = pool.pop()
        parameters = dist._parameters
        for key in sorted(parameters):
            value = parameters[key]
            if not isinstance(value, Distribution):
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
            raise chaospy.StochasticallyDependentError(
                "cycles in dependency structure.")

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


class J(Distribution):
    """
    Joint random variable generator.

    Args:
        args (chaospy.Distribution):
            Distribution to join together.
        rotation (Optional[Sequence[int]]):
            The order of how the joint should be evaluated.
    """

    def __init__(self, *args, **kwargs):
        repr_args = args[:]
        args = [dist for arg in args
                for dist in (arg if isinstance(arg, J) else [arg])]
        assert all(isinstance(dist, Distribution) for dist in args)
        self.interpret_as_integer = all([dist.interpret_as_integer for dist in args])
        self.indices = [0] + numpy.cumsum([len(dist) for dist in args[:-1]]).tolist()
        self.inverse_map = {dist: idx for idx, dist in zip(self.indices, args)}
        dependencies = [
            deps.copy()
            for dist in args
            for deps in dist._dependencies
        ]
        rotation = kwargs.pop("rotation", None)
        assert not kwargs, "'rotation' is the only allowed keyword."

        parameters = {"_%03d" % idx: dist for idx, dist in zip(self.indices, args)}
        super(J, self).__init__(
            parameters=parameters,
            dependencies=dependencies,
            rotation=rotation,
            repr_args=repr_args,
        )

    def _lower(self, cache, **kwargs):
        """
        Example:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
            >>> dist.lower.round(4)
            array([ 0.    , -6.3613])
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> dist.lower
            array([0., 0.])
        """
        uloc = numpy.zeros(len(self))
        for dist in sorted_dependencies(self, cache, reverse=True):
            if dist not in self.inverse_map:
                continue
            idx = self.inverse_map[dist]
            uloc[idx:idx+len(dist)] = dist._get_lower(cache=cache)
        return uloc

    def _upper(self, cache, **kwargs):
        """
        Example:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
            >>> dist.upper.round(4)
            array([1.    , 6.3613])
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> dist.upper
            array([1., 2.])
        """
        uloc = numpy.zeros(len(self))
        for dist in sorted_dependencies(self, cache, reverse=True):
            if dist not in self.inverse_map:
                continue
            idx = self.inverse_map[dist]
            uloc[idx:idx+len(dist)] = dist._get_upper(cache=cache)
        return uloc

    def _cdf(self, xloc, cache, **kwargs):
        """
        Examples:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
            >>> print(dist.fwd([[-0.5, 0.5, 1.5], [-1, 0, 1]]))
            [[0.         0.5        1.        ]
             [0.15865525 0.5        0.84134475]]
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> print(dist.fwd([[-0.5, 0.5, 1.5], [0, 1, 2]]))
            [[0.  0.5 1. ]
             [0.5 0.5 0.5]]
        """
        uloc = numpy.zeros(xloc.shape)
        for dist in sorted_dependencies(self, cache, reverse=True):
            if dist not in self.inverse_map:
                continue
            idx = self.inverse_map[dist]
            uloc[idx:idx+len(dist)] = dist._get_fwd(xloc[idx:idx+len(dist)], cache=cache)[0]
        assert uloc.shape == xloc.shape
        return uloc

    def _pdf(self, xloc, cache, **kwargs):
        """
        Example:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
            >>> dist.pdf([[-0.5, 0.5, 1.5], [-1, 0, 1]]).round(4)
            array([0.    , 0.3989, 0.    ])
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> dist.pdf([[-0.5, 0.5, 1.5], [0, 1, 2]]).round(4)
            array([0., 1., 0.])
        """
        floc = numpy.zeros(xloc.shape)
        for dist in sorted_dependencies(self, cache, reverse=True):
            if dist not in self.inverse_map:
                continue
            idx = self.inverse_map[dist]
            idx = slice(idx, idx+len(dist))
            floc[idx] = dist._get_pdf(xloc[idx], cache=cache)
        return floc

    def _ppf(self, qloc, cache, **kwargs):
        """
        Example:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
            >>> print(numpy.around(dist.inv([[0.1, 0.2, 0.3], [0.3, 0.3, 0.4]]), 4))
            [[ 0.1     0.2     0.3   ]
             [-0.5244 -0.5244 -0.2533]]
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> print(numpy.around(dist.inv([[0.1, 0.2, 0.3], [0.3, 0.3, 0.4]]), 4))
            [[0.1 0.2 0.3]
             [0.4 0.5 0.7]]
        """
        xloc = numpy.zeros(qloc.shape)
        for dist in sorted_dependencies(self, cache, reverse=True):
            if dist not in self.inverse_map:
                continue
            idx = self.inverse_map[dist]
            idx = slice(idx, idx+len(dist))
            xloc[idx] = dist._get_inv(qloc[idx], cache=cache)
        return xloc

    def _mom(self, kloc, cache, **kwargs):
        """
        Example:
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> dist.mom([1, 1]).round(4)
            0.5833
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
            >>> dist.mom([[0, 0, 1], [0, 1, 1]]).round(4)
            array([1., 0., 0.])
        """
        if self.stochastic_dependent:
            raise chaospy.UnsupportedFeature(
                "Joint distribution with dependencies not supported.")
        output = 1.
        for dist in sorted_dependencies(self, cache):
            if dist not in self.inverse_map:
                continue
            idx = self.inverse_map[dist]
            idx = slice(idx, idx+len(dist))
            output *= dist._get_mom(kloc[idx])
        return output

    def _ttr(self, kloc, cache, **kwargs):
        """
        Example:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal(), chaospy.Exponential())
            >>> print(numpy.around(dist.ttr([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), 4))
            [[[0.5    0.5    0.5   ]
              [0.     0.     0.    ]
              [3.     5.     7.    ]]
            <BLANKLINE>
             [[0.0833 0.0667 0.0643]
              [1.     2.     3.    ]
              [1.     4.     9.    ]]]
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> print(numpy.around(dist.ttr([1, 1]), 4)) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
                ...
            chaospy.distributions.baseclass.UnsupportedFeature: Joint ...
        """
        if self.stochastic_dependent:
            raise chaospy.UnsupportedFeature(
                "Joint distribution with dependencies not supported.")
        output = numpy.zeros((2,)+kloc.shape)
        for dist in sorted_dependencies(self, cache):
            if dist not in self.inverse_map:
                continue
            idx = self.inverse_map[dist]
            idx = slice(idx, idx+len(dist))
            alpha, beta = dist._get_ttr(kloc[idx])
            output[0, idx] = alpha
            output[1, idx] = beta
        return output

    def __getitem__(self, i):
        """
        Example:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
            >>> dist[0]
            Uniform()
            >>> dist[1]
            Normal(mu=0, sigma=1)
            >>> dist[:1]
            J(Uniform())
            >>> dist[:2]
            J(Uniform(), Normal(mu=0, sigma=1))
            >>> dist[2]
            Traceback (most recent call last):
                ...
            IndexError: index out of bounds.
        """
        parameters = self.get_parameters(cache={})
        if isinstance(i, int):
            i = "_%03d" % i
            if i in parameters:
                return parameters[i]
            raise IndexError("index out of bounds.")
        if isinstance(i, slice):
            start, stop, step = i.start, i.stop, i.step
            if start is None: start = 0
            if stop is None: stop = len(self)
            if step is None: step = 1
            out = []
            for i in range(start, stop, step):
                out.append(parameters["_%03d" % i])
            return J(*out)
        raise IndexError("index not recognised.")

    def _value(self, cache, **kwargs):
        return self
