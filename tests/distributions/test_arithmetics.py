"""Testing basic distributions and their operations."""
import numpy
import chaospy


def test_dependent_density():
    """Assert that manually create dependency structure holds."""
    distribution1 = chaospy.Exponential(1)
    distribution2 = chaospy.Uniform(lower=0, upper=distribution1)
    distribution = chaospy.J(distribution1, distribution2)
    assert distribution.pdf([0.5, 0.6]) == 0
    assert distribution.pdf([0.5, 0.4]) > 0


def test_distribution_addition(distribution):
    """Assert adding."""
    right_addition = chaospy.E(distribution()+2.0)
    left_addition = chaospy.E(2.0+distribution())
    reference = chaospy.E(distribution())+2.0
    numpy.testing.assert_allclose(right_addition, left_addition, rtol=1e-05, atol=1e-08)
    numpy.testing.assert_allclose(left_addition, reference, rtol=1e-05, atol=1e-08)


def test_distribution_subtraction(distribution):
    """Test distribution subtraction."""
    right_subtraction = chaospy.E(distribution()-3.0)
    left_subtraction = chaospy.E(3.0-distribution())
    reference = chaospy.E(distribution())-3.0
    numpy.testing.assert_allclose(right_subtraction, -left_subtraction, rtol=1e-05, atol=1e-08)
    numpy.testing.assert_allclose(left_subtraction, -reference, rtol=1e-05, atol=1e-08)


def test_distribution_inverse_bounds(distribution):
    """Assert the inverse transformation spans out inside the bounds."""
    distribution = distribution()
    # assert numpy.isclose(distribution.lower, distribution.inv(0))
    # assert numpy.isclose(distribution.upper, distribution.inv(1))
    assert distribution.lower < distribution.inv(0.001)
    assert distribution.inv(0.001) < distribution.inv(0.999)
    assert distribution.inv(0.999) < distribution.upper
