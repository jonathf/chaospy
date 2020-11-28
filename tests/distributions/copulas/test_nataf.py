"""Test for Nataf transformations."""
import numpy
import pytest
import chaospy
from chaospy.distributions.copulas.nataf import nataf


def test_sampling_statistics():
    dists = chaospy.Iid(chaospy.Normal(2, 2), 4)
    corr = numpy.array([[ 1.0, -0.2,  0.3, -0.1],
                        [-0.2,  1.0,  0.0, -0.2],
                        [ 0.3,  0.0,  1.0,  0.4],
                        [-0.1, -0.2,  0.4,  1.0]])
    copula = chaospy.Nataf(dists, corr)
    samples = copula.sample(100000)
    assert numpy.allclose(numpy.mean(samples, axis=-1), chaospy.E(dists), atol=1e-2, rtol=1e-2)
    assert numpy.allclose(numpy.var(samples, axis=-1), chaospy.Var(dists), atol=1e-2, rtol=1e-2)
    assert numpy.allclose(numpy.corrcoef(samples), corr, atol=1e-2, rtol=1e-2)


def test_roundtrip():
    """Assert that forward/inverse Rosenblatt transforms is non-destructive."""
    corr = numpy.array([[ 1.0, -0.9, 0.3],
                        [-0.9,  1.0, 0.0],
                        [ 0.3,  0.0, 1.0]])
    dists = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(1, 2),
                      chaospy.Uniform(2, 3), rotation=[0, 1, 2])
    dist1 = chaospy.Nataf(dists, corr)
    dists = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(1, 2),
                      chaospy.Uniform(2, 3), rotation=[2, 1, 0])
    dist2 = chaospy.Nataf(dists, corr)
    dists = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(1, 2),
                      chaospy.Uniform(2, 3), rotation=[2, 0, 1])
    dist3 = chaospy.Nataf(dists, corr)

    mesh = numpy.mgrid[0.25:0.75:3j, 0.25:0.75:5j, 0.25:0.75:4j].reshape(3, -1)
    assert not numpy.allclose(dist1.fwd(dist2.inv(mesh)), mesh)
    assert numpy.allclose(dist1.fwd(dist1.inv(mesh)), mesh)
    assert numpy.allclose(dist2.fwd(dist2.inv(mesh)), mesh)
    assert numpy.allclose(dist3.fwd(dist3.inv(mesh)), mesh)


def test_rotation():
    """Make sure various rotations do not affect results."""
    corr = numpy.array([[ 1.0, -0.2, 0.3],
                        [-0.2,  1.0, 0.0],
                        [ 0.3,  0.0, 1.0]])

    dists = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(1, 2),
                      chaospy.Uniform(2, 3), rotation=[0, 1, 2])
    dist1 = chaospy.Nataf(dists, corr)
    dists = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(1, 2),
                      chaospy.Uniform(2, 3), rotation=[2, 1, 0])
    dist2 = chaospy.Nataf(dists, corr)
    dists = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(1, 2),
                      chaospy.Uniform(2, 3), rotation=[2, 0, 1])
    dist3 = chaospy.Nataf(dists, corr)

    mesh = numpy.mgrid[0.1:0.9:2j, 1.1:1.9:3j, 2.1:2.9:4j]

    # PDF should be unaffected by rotation.
    assert numpy.allclose(dist1.pdf(mesh), dist2.pdf(mesh))
    assert numpy.allclose(dist1.pdf(mesh), dist3.pdf(mesh))
    assert numpy.allclose(dist2.pdf(mesh), dist3.pdf(mesh))

    # One axis is constant. Verify that is the case for each rotation.
    assert numpy.all(dist1.fwd(mesh)[0, :, 0, 0] == dist1.fwd(mesh)[0].T)
    assert numpy.all(dist2.fwd(mesh)[2, 0, 0, :] == dist2.fwd(mesh)[2])
    assert numpy.all(dist3.fwd(mesh)[2, 0, 0, :] == dist2.fwd(mesh)[2])


def test_sampling():
    """Assert that inverse mapping results in samples with the correct statistics."""
    mean = numpy.array([100, 10, 50])
    cov = numpy.array([[ 10, -8,  4],
                       [ -8, 20,  2],
                       [  4,  2, 30]])
    samples_u = chaospy.Uniform().sample((3, 100000))
    for rotation in [(0, 1, 2), (2, 1, 0), (2, 0, 1)]:
        dist = chaospy.MvNormal(mean, cov, rotation=rotation)
        samples_ = dist.inv(samples_u)
        mean_ = numpy.mean(samples_, axis=-1)
        assert numpy.allclose(mean_, mean, atol=1e-3, rtol=1e-3)
        cov_ = numpy.cov(samples_)
        assert numpy.allclose(cov_, cov, atol=1e-2, rtol=1e-2)


def test_segmented_mappings():
    """Assert that conditional distributions in various rotation works as expected."""
    correlation = numpy.array([[1, 0.4, -0.5], [0.4, 1, 0], [-0.5, 0, 1]])

    for rot in [(0, 1, 2), (2, 1, 0), (2, 0, 1)]:

        copula = nataf(correlation, rotation=rot)
        samples = copula.sample(100)
        isamples = copula.fwd(samples)
        assert numpy.all(isamples < 1) and numpy.all(isamples > 0)
        density = copula.pdf(samples, decompose=True)

        caches = [{}, {(rot[0], copula): (samples[rot[0]], isamples[rot[0]])},
                  {(rot[0], copula): (samples[rot[0]], isamples[rot[0]]),
                   (rot[1], copula): (samples[rot[1]], isamples[rot[1]])}]
        for idx, cache in zip(rot, caches):
            assert numpy.allclose(copula._get_fwd(samples[idx], idx, cache=cache.copy()), isamples[idx])
            assert numpy.allclose(copula._get_inv(isamples[idx], idx, cache=cache.copy()), samples[idx])
            if cache:
                with pytest.raises(chaospy.StochasticallyDependentError):
                    copula[idx].fwd(samples[idx])
                with pytest.raises(chaospy.StochasticallyDependentError):
                    copula[idx].inv(isamples[idx])
                with pytest.raises(chaospy.StochasticallyDependentError):
                    copula[idx].pdf(samples[idx])
            else:
                assert numpy.allclose(copula[idx].fwd(samples[idx]), isamples[idx])
                assert numpy.allclose(copula[idx].inv(isamples[idx]), samples[idx])
                assert numpy.allclose(copula[idx].pdf(samples[idx]), density[idx])

            with pytest.raises(chaospy.StochasticallyDependentError):
                copula[idx].ttr(0)
            with pytest.raises(chaospy.UnsupportedFeature):
                copula[idx].mom(1, allow_approx=False)


def test_slicing():
    """Test if slicing of distribution works as expected."""
    dist = chaospy.MvNormal(mu=[1, 2], sigma=[[4, -1], [-1, 3]], rotation=[1, 0])
    with pytest.raises(IndexError):
        dist[-2]
    with pytest.raises(IndexError):
        dist[2]
    with pytest.raises(IndexError):
        dist["illigal_index"]
    assert dist[-1] == dist[1]
    assert dist[1].mom(1) == 2
    # assert dist[:1] == dist[0]
