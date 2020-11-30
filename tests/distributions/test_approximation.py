"""Test if approximation functions works as expected."""
import numpy
import chaospy
from chaospy.distributions.collection.gamma import gamma

def fail_function(*args, **kwargs):
    """Function causing failure."""
    raise chaospy.UnsupportedFeature()

DIST = gamma(1)
SAMPLES = DIST.sample(100)


def test_approximate_density(monkeypatch):
    """Assert that approximate density is doing its job."""
    ref_density = DIST.pdf(SAMPLES)
    monkeypatch.setattr(DIST, "_pdf", fail_function)
    assert numpy.allclose(DIST.pdf(SAMPLES, allow_approx=True), ref_density)


def test_approximate_inverse(monkeypatch):
    """Assert that approximate inverse is doing its job."""
    u_samples = DIST.fwd(SAMPLES)
    monkeypatch.setattr(DIST, "_ppf", fail_function)
    assert numpy.allclose(DIST.inv(u_samples), SAMPLES)


def test_approximate_moment(monkeypatch):
    """Assert that approximate moments is doing its job."""
    ref_moments = DIST.mom([1, 2, 3, 4])
    monkeypatch.setattr(DIST, "_mom", fail_function)
    assert numpy.allclose(DIST.mom([1, 2, 3, 4]), ref_moments)
