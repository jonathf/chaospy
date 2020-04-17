"""Test interface against numpoly."""
import chaospy


def test_constant_expected():
    """Test if polynomial constant behave as expected."""
    distribution = chaospy.J(chaospy.Uniform(-1.2, 1.2),
                             chaospy.Uniform(-2.0, 2.0))
    const = chaospy.polynomial(7.)
    assert chaospy.E(const, distribution[0]) == const
    assert chaospy.E(const, distribution) == const
    assert chaospy.Var(const, distribution) == 0.
