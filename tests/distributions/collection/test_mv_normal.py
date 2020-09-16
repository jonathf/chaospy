"""Test for multivariate normal distribution."""
import numpy
import pytest
import chaospy


def test_sampling_statistics():
    """Assert that given mean and covariance matches sample statistics."""
    mean = [100, 10, 10, 100]
    cov = [[10, -9,  3, -1],
           [-9, 20,  1, -2],
           [ 3,  1, 30,  4],
           [-1, -2,  4, 40]]
    dist = chaospy.MvNormal(mean, cov, rotation=[3, 0, 2, 1])
    samples = dist.sample(100000)
    assert numpy.allclose(
        numpy.mean(samples, axis=-1), mean, atol=1e-2, rtol=1e-3), numpy.mean(samples, axis=-1).round(3)
    assert numpy.allclose(
        numpy.cov(samples), cov, atol=1e-1, rtol=1e-2), numpy.cov(samples).round(3)


def test_roundtrip():
    """Assert that forward/inverse Rosenblatt transforms is non-destructive."""
    mean = numpy.array([0.1, -0.5, 0])
    cov = numpy.array([[ 1.0, -0.9, 0.3],
                       [-0.9,  1.0, 0.0],
                       [ 0.3,  0.0, 1.0]])
    dist1 = chaospy.MvNormal(mean, cov, rotation=[0, 1, 2])
    dist2 = chaospy.MvNormal(mean, cov, rotation=[2, 1, 0])
    dist3 = chaospy.MvNormal(mean, cov, rotation=[2, 0, 1])

    mesh = numpy.mgrid[0.25:0.75:3j, 0.25:0.75:5j, 0.25:0.75:4j].reshape(3, -1)
    assert numpy.allclose(dist1.fwd(dist1.inv(mesh)), mesh)
    assert numpy.allclose(dist2.fwd(dist2.inv(mesh)), mesh)
    assert numpy.allclose(dist3.fwd(dist3.inv(mesh)), mesh)


def test_rotation():
    """Make sure various rotations do not affect results."""
    mean = numpy.array([0, 10, 0])
    cov = numpy.array([[ 1.0, -0.2, 0.3],
                       [-0.2,  1.0, 0.0],
                       [ 0.3,  0.0, 1.0]])
    mesh = numpy.mgrid[-1:1:2j, 9:11:2j, -1:1:2j]
    dist1 = chaospy.MvNormal(mean, cov, rotation=[0, 1, 2])
    dist2 = chaospy.MvNormal(mean, cov, rotation=[2, 1, 0])
    dist3 = chaospy.MvNormal(mean, cov, rotation=[2, 0, 1])

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


def test_dependencies_1d():
    """Assert that mu being a 1-D distribution is fine."""
    mu = chaospy.Uniform(0, 1)
    dist = chaospy.MvNormal(mu=mu, sigma=[[1, 0.5], [0.5, 1]])
    joint = chaospy.J(mu, dist)

    samples = joint.sample(10000)
    mean = numpy.array([1/2., 1/2., 1/2.])
    covariance = numpy.array([[1/12.,  1/12.,  1/12.],
                              [1/12., 13/12.,  7/12.],
                              [1/12.,  7/12., 13/12.]])
    assert numpy.allclose(numpy.mean(samples, axis=-1), mean, rtol=1e-2, atol=1e-2)
    assert numpy.allclose(numpy.cov(samples), covariance, rtol=1e-2, atol=1e-2)


def test_dependencies_2d():
    """Assert that mu being a 2-D distribution is fine."""
    mu = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(1, 2))
    dist = chaospy.MvNormal(mu=mu, sigma=[[1, 0.5], [0.5, 1]])
    joint = chaospy.J(mu, dist)

    samples = joint.sample(10000)
    mean = numpy.array([1/2., 3/2., 1/2., 3/2.])
    covariance = numpy.array([[1/12.,    0.,  1/12.,     0.],
                              [   0., 1/12.,      0,  1/12.],
                              [1/12.,    0., 13/12.,   1/2.],
                              [   0., 1/12.,   1/2., 13/12.]])
    assert numpy.allclose(numpy.mean(samples, axis=-1), mean, rtol=1e-2, atol=1e-2)
    assert numpy.allclose(numpy.cov(samples), covariance, rtol=1e-2, atol=1e-2)


def test_segmented_mappings():
    """Assert that conditional distributions in various rotation works as expected."""
    mean_ref = numpy.array([1, 10, 100])
    covariance = numpy.array([[1, 0.4, -0.5], [0.4, 2, 0], [-0.5, 0, 3]])

    for rot in [(0, 1, 2), (2, 1, 0), (2, 0, 1)]:
        dist = chaospy.MvNormal(mean_ref, covariance, rotation=rot)
        mean = mean_ref[numpy.array(rot)]
        samples = dist.sample(100)
        isamples = dist.fwd(samples)
        density = dist.pdf(samples, decompose=True)

        dependencies = [(), chaospy.J(dist[rot[0]]), chaospy.J(dist[rot[0]], dist[rot[1]])]
        caches = [{}, {dist[rot[0]]: samples[rot[0]], rot[0]: isamples[rot[0]]},
                  {dist[rot[0]]: samples[rot[0]], rot[0]: isamples[rot[0]],
                   dist[rot[1]]: samples[rot[1]], rot[1]: isamples[rot[1]]}]
        for idx, deps, cache in zip(rot, dependencies, caches):
            assert numpy.allclose(dist[idx]._cdf(samples[idx], idx, dist, deps, cache=cache), isamples[idx])
            assert numpy.allclose(dist[idx]._ppf(isamples[idx], idx, dist, deps, cache=cache), samples[idx])
            assert numpy.allclose(dist[idx]._pdf(samples[idx], idx, dist, deps, cache=cache), density[idx])


def test_slicing():
    """Test if slicing of distribution works as expected."""
    dist = chaospy.MvNormal(mu=[0, 0], sigma=[[1, 0.5], [0.5, 1]])
    with pytest.raises(IndexError):
        dist[-2]
    with pytest.raises(IndexError):
        dist[2]
    assert dist[-1] == dist[1]
    # assert dist[:1] == dist[0]
