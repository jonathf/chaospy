"""Test basic properties with the Dist baseclass."""
import pytest
import chaospy


def test_distribution_exclusion():
    """
    Check if illegal reuse of dependencies raises errors correctly.

    Dependencies transformed in a non-bijective way can not be reused. For
    example, here a truncation of a distribution can not be use together with
    said distribution without truncation.
    """
    dist1 = chaospy.Uniform(-1, 1)
    dist2 = chaospy.Trunc(dist1, 0)
    with pytest.raises(chaospy.StochasticallyDependentError):
        dist3 = chaospy.J(dist1, dist2)


def test_incomplete_stochastic_dependency():
    """
    Check if dangling stochastic dependency raises errors correctly.

    Many operators requires that the number of underlying distributions
    is the same as the length of the stochastic vector.
    """
    dist1 = chaospy.Uniform(0, 1)
    dist2 = chaospy.Normal(dist1, 1)
    with pytest.raises(chaospy.StochasticallyDependentError):
        dist2.pdf(0)
    with pytest.raises(chaospy.StochasticallyDependentError):
        dist2.fwd(0)
    with pytest.raises(chaospy.StochasticallyDependentError):
        dist2.cdf(0)
    with pytest.raises(chaospy.StochasticallyDependentError):
        dist2.inv(0.5)


def test_underdefined_distribution():
    """
    Check if under-defined probability distributions raises errors correctly.

    The number of underlying distribution components in a stochastic vector
    must always be at least as large as the length of the vector.
    """
    with pytest.raises(chaospy.StochasticallyDependentError):
        chaospy.Add(2, 2)
    dist = chaospy.Uniform(-1, 1)
    with pytest.raises(chaospy.StochasticallyDependentError):
        chaospy.J(dist, dist)
