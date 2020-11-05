"""Check if Stieltjes method, both analytical and discretized works as expected."""
import numpy
import numpoly
import chaospy


def test_analytical_stieltjes(analytical_distribution):
    """Assert that Analytical Stieltjes produces orthogonality."""
    coeffs, [orth], norms = chaospy.analytical_stieltjes(
        order=4, dist=analytical_distribution)
    assert orth[0] == 1
    assert numpy.allclose(chaospy.E(orth[1:], analytical_distribution), 0)
    covariance = chaospy.E(
        numpoly.outer(orth[1:], orth[1:]), analytical_distribution)
    assert numpy.allclose(numpy.diag(numpy.diag(covariance)), covariance)
    assert numpy.allclose(numpoly.lead_coefficient(orth), 1)


def test_stieltjes_compared(analytical_distribution):
    """Assert that discretized and analytical approach are equivalent."""
    (alpha0, beta0), [orth0], norms0 = chaospy.analytical_stieltjes(
        order=3, dist=analytical_distribution)
    (alpha1, beta1), [orth1], norms1 = chaospy.discretized_stieltjes(
        order=3, dist=analytical_distribution)
    assert numpy.allclose(alpha0, alpha1)
    assert numpy.allclose(beta0, beta1)
    assert numpy.allclose(orth0.coefficients, orth1.coefficients)
    assert numpy.allclose(norms0, norms1)
