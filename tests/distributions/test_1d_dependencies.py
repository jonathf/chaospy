"""Test dependent distributions with 1-D components."""
from pytest import raises
import numpy
import chaospy

DIST1 = chaospy.Uniform(1, 2)
DIST2 = chaospy.Gamma(DIST1)
JOINT1 = chaospy.J(DIST1, DIST2)
JOINT2 = chaospy.J(DIST2, DIST1)


def test_1d_stochastic_dependencies():
    """Ensure ``stochastic_dependencies`` behaves as expected for dependent 1-D distributions."""
    assert not DIST1.stochastic_dependent
    assert DIST2.stochastic_dependent
    assert JOINT1.stochastic_dependent
    assert JOINT2.stochastic_dependent


def test_1d_dependent_bounds():
    """Ensure lower and upper bounds works for dependent 1-D distributions."""
    assert numpy.isclose(DIST2.lower, 0)
    assert numpy.isclose(DIST2.upper, 35.84367486)
    assert numpy.allclose(JOINT1.lower, [1, 0])
    assert numpy.allclose(JOINT1.upper, [2, 35.84367486])
    assert numpy.allclose(JOINT2.lower, [0, 1])
    assert numpy.allclose(JOINT2.upper, [35.84367486, 2])


def test_1d_dependent_mapping():
    """Ensure inverse and forward behaves as expected for dependent 1-D distributions."""
    grid = numpy.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    inv_map1 = numpy.array([[1, 1, 2, 2], [0, 32.2369909, 0, 35.84367486]])
    inv_map2 = numpy.array([[0, 0, 32.2369909, 35.84367486], [1, 2, 1, 2]])

    assert numpy.allclose(JOINT1.inv(grid), inv_map1)
    assert numpy.allclose(JOINT2.inv(grid), inv_map2)
    assert numpy.allclose(JOINT1.fwd(inv_map1), grid)
    assert numpy.allclose(JOINT2.fwd(inv_map2), grid)


def test_1d_dependent_density():
    """Ensure probability density function behaves as expected for dependent 1-D distributions."""
    x_loc1 = numpy.array([0.8, 1.8, 1.2, 1.8])
    x_loc2 = numpy.array([2, 4, 6, 8])
    y_loc1 = numpy.array([0, 1, 1, 1])
    y_loc2 = numpy.array([0.1011967, 0.05961306, 0.00386314, 0.00190102])
    assert numpy.allclose(
        JOINT1.pdf([x_loc1, x_loc2], decompose=True, allow_approx=False),
        [y_loc1, y_loc2]
    )
    assert numpy.allclose(
        JOINT2.pdf([x_loc2, x_loc1], decompose=True, allow_approx=False),
        [y_loc2, y_loc1]
    )
