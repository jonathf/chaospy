"""Tests for the various base-operators."""
import numpy
import chaospy


def test_operator_slicing():
    dists = [
        chaospy.J(chaospy.Normal(2), chaospy.Normal(2), chaospy.Normal(3)),  # ShiftScale
        chaospy.J(chaospy.Uniform(1, 3), chaospy.Uniform(1, 3), chaospy.Uniform(1, 5)),  # LowerUpper
        chaospy.MvNormal([2, 2, 3], numpy.eye(3)),  # MeanCovariance
    ]
    for dist in dists:

        assert numpy.allclose(dist.inv([0.5, 0.5, 0.5]), [2, 2, 3])
        assert numpy.allclose(dist.fwd([2, 2, 3]), [0.5, 0.5, 0.5])
        density = dist.pdf([2, 2, 2], decompose=True)
        assert numpy.isclose(density[0], density[1]), dist
        assert not numpy.isclose(density[0], density[2]), dist
        assert numpy.isclose(dist[0].inv([0.5]), 2)
        assert numpy.isclose(dist[0].fwd([2]), 0.5)
