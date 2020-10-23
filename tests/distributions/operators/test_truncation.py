"""Tests for truncation operator."""
import numpy
import chaospy


def test_truncation_lower_as_dist():
    """Ensure lower bound as a distribution is supported."""
    dist1 = chaospy.Normal()
    dist2 = chaospy.Trunc(chaospy.Normal(), lower=dist1)
    joint = chaospy.J(dist1, dist2)
    ref10 = (0.5-dist1.fwd(-1))/(1-dist1.fwd(-1))
    assert numpy.allclose(joint.fwd([[-1, 0, 1], [0, 0, 0]]),
                          [dist1.fwd([-1, 0, 1]), [ref10, 0, 0]])


def test_truncation_upper_as_dist():
    """Ensure upper bound as a distribution is supported."""
    dist1 = chaospy.Normal()
    dist2 = chaospy.Trunc(chaospy.Normal(), upper=dist1)
    joint = chaospy.J(dist1, dist2)
    ref12 = 0.5/dist1.fwd(1)
    assert numpy.allclose(joint.fwd([[-1, 0, 1], [0, 0, 0]]),
                          [dist1.fwd([-1, 0, 1]), [1, 1, ref12]])


def test_truncation_both_as_dist():
    """Ensure that lower and upper bound combo is supported."""
    dist1 = chaospy.Normal()
    dist2 = chaospy.Normal()
    dist3 = chaospy.Trunc(chaospy.Normal(), lower=dist1, upper=dist2)
    joint = chaospy.J(dist1, dist2, dist3)
    ref21 = (0.5-dist1.fwd(-1))/(1-dist1.fwd(-1))/dist2.fwd(1)
    assert numpy.allclose(joint.fwd([[-1, -1,  1,  1],
                                     [-1,  1, -1,  1],
                                     [ 0,  0,  0,  0]]),
                          [dist1.fwd([-1, -1,  1,  1]),
                           dist2.fwd([-1,  1, -1,  1]),
                           [1, ref21, 1, 0]])


def test_trucation_multivariate():
    """Ensure that multivariate bounds works as expected."""
    dist1 = chaospy.Iid(chaospy.Normal(), 2)
    dist2 = chaospy.Trunc(chaospy.Iid(chaospy.Normal(), 2),
                          lower=dist1, upper=[1, 1])
    joint = chaospy.J(dist1, dist2)
    assert numpy.allclose(
        joint.fwd([[-1, -1, -1, -1],
                   [-1, -1, -1, -1],
                   [ 0,  0, -2,  2],
                   [-2,  2,  0,  0]]),
        [[0.15865525, 0.15865525, 0.15865525, 0.15865525],
         [0.15865525, 0.15865525, 0.15865525, 0.15865525],
         [0.48222003, 0.48222003, 0.        , 1.        ],
         [0.        , 1.        , 0.48222003, 0.48222003]],
    )

