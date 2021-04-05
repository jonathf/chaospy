"""Test dependent distributions with 2-D components."""
from pytest import raises
import numpy
import chaospy

DIST1 = chaospy.J(chaospy.Uniform(1, 2), chaospy.Uniform(2, 4))
DIST2 = chaospy.J(chaospy.Gamma(DIST1[0]), chaospy.Gamma(DIST1[1]))
JOINT1 = chaospy.J(DIST1, DIST2)
JOINT2 = chaospy.J(DIST2, DIST1)


def test_2d_stochastic_dependencies():
    """Ensure ``stochastic_dependencies`` behaves as expected for dependent 2-D distributions."""
    assert not DIST1.stochastic_dependent
    assert DIST2.stochastic_dependent
    assert JOINT1.stochastic_dependent
    assert JOINT2.stochastic_dependent


def test_2d_dependencies():
    """Ensure 2-D dependencies behaves as expected."""
    grid1 = numpy.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    grid2 = numpy.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1]])
    inv_map1 = numpy.array([[1, 1, 2, 2],
                            [2, 4, 2, 4],
                            [32.2369909,  32.2369909,  35.84367486, 35.84367486],
                            [35.84367486, 41.71021463, 35.84367486, 41.71021463]])
    inv_map2 = numpy.array([[32.2369909,  32.2369909,  35.84367486, 35.84367486],
                            [35.84367486, 41.71021463, 35.84367486, 41.71021463],
                            [1, 1, 2, 2],
                            [2, 4, 2, 4]])
    assert numpy.allclose(JOINT1.inv(grid1), inv_map1)
    assert numpy.allclose(JOINT2.inv(grid2), inv_map2)


def test_2d_dependent_density():
    """Ensure probability density function behaves as expected for dependent 2-D distributions."""
    x_loc1 = numpy.array([0.8, 1.8, 1.2, 1.8])
    x_loc2 = numpy.array([1.8, 3.8, 3.2, 3.8])
    x_loc3 = numpy.array([2, 4, 6, 8])
    x_loc4 = numpy.array([2, 4, 6, 8])

    y_loc1 = numpy.array([0, 1, 1, 1])
    y_loc2 = numpy.array([0, 0.5, 0.5, 0.5])
    y_loc3 = numpy.array([0.1011967, 0.05961306, 0.00386314, 0.00190102])
    y_loc4 = numpy.array([0.25299175, 0.1892478, 0.05267923, 0.02413998])

    assert numpy.allclose(
        JOINT1.pdf([x_loc1, x_loc2, x_loc3, x_loc4], decompose=True, allow_approx=False),
        [y_loc1, y_loc2, y_loc3, y_loc4]
    )
    assert numpy.allclose(
        JOINT2.pdf([x_loc3, x_loc4, x_loc1, x_loc2], decompose=True, allow_approx=False),
        [y_loc3, y_loc4, y_loc1, y_loc2]
    )
