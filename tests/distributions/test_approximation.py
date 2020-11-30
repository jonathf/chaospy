"""Test if approximation functions works as expected."""
from pytest import raises
import numpy
import chaospy
from chaospy.distributions.collection.gamma import gamma

DIST = gamma(1)
SAMPLES = DIST.sample(100)


def test_approximate_density(monkeypatch):
    """Assert that approximate density is doing its job."""
    ref_density = DIST.pdf(SAMPLES)
    monkeypatch.setattr(DIST, "_pdf",
                        lambda x, **_: chaospy.Distribution._pdf(DIST, x))
    with raises(chaospy.UnsupportedFeature):
        DIST.pdf(SAMPLES, allow_approx=False)
    assert numpy.allclose(DIST.pdf(SAMPLES, allow_approx=True), ref_density)


def test_approximate_inverse(monkeypatch):
    """Assert that approximate inverse is doing its job."""
    u_samples = DIST.fwd(SAMPLES)
    monkeypatch.setattr(DIST, "_ppf",
                        lambda u, **_: chaospy.Distribution._ppf(DIST, u))
    assert numpy.allclose(DIST.inv(u_samples), SAMPLES)


def test_approximate_moment(monkeypatch):
    """Assert that approximate moments is doing its job."""
    ref_moments = DIST.mom([1, 2, 3, 4])
    monkeypatch.setattr(DIST, "_mom",
                        lambda k, **_: chaospy.Distribution._mom(DIST, k))
    monkeypatch.delitem(DIST._mom_cache, (1,))  # value is cached
    with raises(chaospy.UnsupportedFeature):
        DIST.mom([1, 2, 3, 4], allow_approx=False)
    assert numpy.allclose(DIST.mom([1, 2, 3, 4], allow_approx=True), ref_moments)
