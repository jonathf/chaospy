"""
There are three ways to create a multivariate probability distribution in
``chaospy``: Using the joint constructor
:class:`~chaospy.distributions.operators.joint.J`, the identical independent
distribution constructor: :class:`~chaospy.distributions.operators.iid.Iid`,
and to one of the pre-constructed multivariate distribution defined in
:ref:`listdistributions`.

Constructing a multivariate probability distribution can be done using the
:func:`~chaospy.distributions.operators.joint.J` constructor. E.g.::

    >>> distribution = chaospy.J(chaospy.Normal(), chaospy.Uniform())
    >>> distribution
    J(Normal(mu=0, sigma=1), Uniform())

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
import logging
import numpy
import chaospy

from ..baseclass import Distribution


class J(Distribution):
    """
    Joint random variable.

    Args:
        args (chaospy.Distribution):
            Distribution to join together.
        rotation (Optional[Sequence[int]]):
            The order of how the joint should be evaluated.
    """

    def __init__(self, *args, **kwargs):
        dists = []
        repr_args = []
        owners = {}
        index = []
        dependencies = []
        idx = 0
        for arg in args:
            assert isinstance(arg, Distribution)
            if isinstance(arg, J):
                repr_args.extend(arg._repr_args)
                owners.update({idx+idx_: value
                               for idx_, value in arg._owners.items()})
                index += list(range(idx, idx+len(args)))
                idx += len(arg)
            elif len(arg) > 1:
                repr_args.append(arg)
                owners.update({idx2: (idx1, arg)
                               for idx1, idx2 in enumerate(range(idx, idx+len(arg)))})
                index += [idx]*len(arg)
                idx += 1
            else:
                repr_args.append(arg)
                owners[idx] = (0, arg)
                index += [idx]
                idx += 1
            dependencies += [dep.copy() for dep in arg._dependencies]

        self.interpret_as_integer = all([dist.interpret_as_integer
                                         for (_, dist) in owners.values()])
        rotation = kwargs.pop("rotation", None)
        assert not kwargs, "'rotation' is the only allowed keyword."
        parameters = {"_%03d" % idx: dist for idx, dist in enumerate(repr_args)}
        parameters["index"] = numpy.asarray(index)

        super(J, self).__init__(
            parameters=parameters,
            dependencies=dependencies,
            rotation=rotation,
            repr_args=repr_args,
        )
        self._owners = owners

    def get_parameters(self, idx, cache, assert_numerical=True):
        del assert_numerical  # joint is never numerical on its own.
        parameters = super(J, self).get_parameters(
            idx, cache, assert_numerical=False)
        if idx is None:
            return dict(index=parameters["index"])
        idx, dist = self._owners[idx]
        return dict(idx=idx, dist=dist, cache=cache)

    def _lower(self, idx, dist, cache):
        """
        Example:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
            >>> dist.lower.round(4)
            array([ 0.  , -8.22])
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> dist.lower
            array([0., 0.])
        """
        return dist._get_lower(idx, cache)

    def _upper(self, idx, dist, cache):
        """
        Example:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
            >>> dist.upper.round(4)
            array([1.  , 8.22])
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> dist.upper
            array([1., 2.])
        """
        return dist._get_upper(idx, cache)

    def _cdf(self, xloc, idx, dist, cache):
        """
        Examples:
            >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
            >>> dist.fwd([[0, 0.5, 1], [-1, 0, 1]]).round(4)
            array([[0.    , 0.5   , 1.    ],
                   [0.1587, 0.5   , 0.8413]])
            >>> d0 = chaospy.Uniform()
            >>> dist = chaospy.J(d0, d0+chaospy.Uniform())
            >>> dist.fwd([[0, 0.5, 1], [0.5, 1, 1.5]]).round(4)
            array([[0. , 0.5, 1. ],
                   [0.5, 0.5, 0.5]])

        """
        return dist._get_fwd(xloc, idx, cache)

    def _pdf(self, xloc, idx, dist, cache):
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
        return dist._get_pdf(xloc, idx, cache)

    def _ppf(self, qloc, idx, dist, cache):
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
        return dist._get_inv(qloc, idx, cache)

    def _mom(self, kloc, index):
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
        output = 1.
        for idx1 in range(len(self._owners)):
            _, dist1 = self._owners[idx1]
            for idx2 in range(idx1+1, len(self._owners)):
                _, dist2 = self._owners[idx2]
                if chaospy.shares_dependencies(dist1, dist2):
                    raise chaospy.UnsupportedFeature(
                        "Shared dependencies across joint")

        kloc = kloc[self._rotation]
        for unique_idx in numpy.unique(index[self._rotation]):
            output *= self._owners[unique_idx][1]._get_mom(kloc[index == unique_idx])
        return output

    def _ttr(self, kloc, idx, dist, cache):
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
        return dist._get_ttr(kloc, idx)

    def __getitem__(self, index):
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
            IndexError: list index out of range
        """
        if isinstance(index, int):
            if index <= -len(self) or index >= len(self):
                raise IndexError("list index out of range")
            return self.get_parameters(index, cache={}, assert_numerical=False)["dist"]
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = len(self) if index.stop is None else index.stop
            step = 1 if index.step is None else index.step
            index = range(start, stop, step)
        return J(*[self[idx] for idx in index])

    def _cache(self, idx, cache):
        if idx is None:
            return self
        parameters = self.get_parameters(idx, cache, assert_numerical=False)
        out = parameters["dist"]._get_cache(parameters["idx"], cache)
        if isinstance(out, chaospy.Distribution):
            return self
        return out


# def J(*args, **kwargs):
#     """
#     Joint random variable.

#     Too be deprecated, use `chaospy.J` instead.

#     Args:
#         args (chaospy.Distribution):
#             Distribution to join together.
#     """
#     logger = logging.getLogger(__name__)
#     logger.warning("DepricationWarning: J to be replaced with Joint.")
#     return Joint(*args, **kwargs)
