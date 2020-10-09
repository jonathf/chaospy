"""Independent identical distributed vector of random variables."""
from copy import deepcopy
import numpy
import chaospy

from ..baseclass import Distribution
from .joint import J


class Iid(J):
    """
    Opaque method for creating independent identical distributed random
    variables from an univariate variable.

    Args:
        dist (Distribution):
            Distribution to make into i.i.d. vector. The
            distribution will be copied so to not become a part of
            the dependency graph.
        length (int):
            The length of new distribution.

    Examples:
        >>> distribution = chaospy.Iid(chaospy.Normal(0, 1), 2)
        >>> distribution
        Iid(Normal(mu=0, sigma=1), 2)
        >>> chaospy.Cov(distribution)
        array([[1., 0.],
               [0., 1.]])
        >>> mesh = numpy.mgrid[0.25:0.75:3j, 0.25:0.75:3j].reshape(2, -1)
        >>> mapped_mesh = distribution.inv(mesh)
        >>> mapped_mesh.round(2)
        array([[-0.67, -0.67, -0.67,  0.  ,  0.  ,  0.  ,  0.67,  0.67,  0.67],
               [-0.67,  0.  ,  0.67, -0.67,  0.  ,  0.67, -0.67,  0.  ,  0.67]])
        >>> distribution.fwd(mapped_mesh).round(2)
        array([[0.25, 0.25, 0.25, 0.5 , 0.5 , 0.5 , 0.75, 0.75, 0.75],
               [0.25, 0.5 , 0.75, 0.25, 0.5 , 0.75, 0.25, 0.5 , 0.75]])
        >>> distribution.pdf(mapped_mesh).round(3)
        array([0.101, 0.127, 0.101, 0.127, 0.159, 0.127, 0.101, 0.127, 0.101])
        >>> distribution.sample(4, rule="halton").round(3)
        array([[-1.15 ,  0.319, -0.319,  1.15 ],
               [-0.14 ,  0.765, -0.765,  0.14 ]])
        >>> distribution.mom([[1, 2, 2], [2, 1, 2]]).round(12)
        array([0., 0., 1.])

    """

    def __init__(self, dist, length, rotation=None):
        assert isinstance(dist, Distribution)
        assert len(dist) == 1 and length >= 1
        assert len(dist._dependencies[0]) == 1
        exclusion = dist._dependencies[0].copy()
        dists = [deepcopy(dist) for _ in range(length)]
        for dist in dists:
            dist._dependencies = chaospy.init_dependencies(
                dist, rotation=[0], dependency_type="iid")
        super(Iid, self).__init__(*dists, rotation=rotation)
        self._exclusion.update(exclusion)
        self._repr_args = [dist, length]
