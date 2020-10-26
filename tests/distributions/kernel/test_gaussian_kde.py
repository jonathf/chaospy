"""Tests for Gaussian kernel density estimation."""
import numpy
import chaospy


def test_gaussian_kde_1d_integration():
    """Make sure that 1D distribution integration is correct."""
    dist = chaospy.GaussianKDE([0, 1, 2])
    t = numpy.mgrid[-2.6:4.6:2e5j]
    scale = numpy.ptp(t)
    assert numpy.isclose(numpy.mean(dist.pdf(t)*scale), 1.)
    assert numpy.isclose(numpy.mean(t*dist.pdf(t)*scale), 1.)


def test_gaussian_kde_2d_integration():
    """Make sure that 2D distribution integration is correct."""
    dist = chaospy.GaussianKDE([[0, 2], [2, 0]])
    samples = dist.sample(1e4)
    assert numpy.allclose(numpy.mean(samples, axis=-1), 1, rtol=1e-1)


def test_gaussian_kde_rotation():
    """Make sure rotation does not affect mapping."""
    dist = chaospy.GaussianKDE([[0, 0, 2], [0, 2, 0], [2, 0, 0]], rotation=[0, 1, 2])
    grid = numpy.mgrid[0.01:0.99:2j, 0.01:0.99:2j, 0.01:0.99:2j].reshape(3, 8)
    inverse = dist.inv(grid)
    assert numpy.allclose(dist.fwd(inverse), grid)
    assert numpy.allclose(dist.pdf(inverse),
                          [2.550e-05, 2.553e-05, 2.553e-05, 2.552e-05,
                           2.525e-05, 2.522e-05, 2.522e-05, 2.519e-05])
    assert numpy.allclose(
        inverse,
        [[-1.38424, -1.38424, -1.38424, -1.38424,  3.19971,  3.19971,  3.19971,  3.19971],
         [-1.31003, -1.31003,  3.31003,  3.31003, -1.48391, -1.48391,  1.48429,  1.48429],
         [ 0.51571,  3.48391, -1.48391,  1.48413, -1.48391,  1.48429, -1.48391,  1.48429]],
        rtol=1e-5,
    )
    dist = chaospy.GaussianKDE([[0, 0, 2], [0, 2, 0], [2, 0, 0]], rotation=[2, 1, 0])
    inverse = dist.inv(grid)
    assert numpy.allclose(dist.fwd(inverse), grid)
    assert numpy.allclose(dist.pdf(inverse),
                          [2.550e-05, 2.525e-05, 2.553e-05, 2.522e-05,
                           2.553e-05, 2.522e-05, 2.552e-05, 2.519e-05])
    assert numpy.allclose(
        inverse,
        [[ 0.51571, -1.48391, -1.48391, -1.48391,  3.48391,  1.48429,  1.48413,  1.48429],
         [-1.31003, -1.48391,  3.31003,  1.48429, -1.31003, -1.48391,  3.31003,  1.48429],
         [-1.38424,  3.19971, -1.38424,  3.19971, -1.38424,  3.19971, -1.38424,  3.19971]],
        rtol=1e-5,
    )
    dist = chaospy.GaussianKDE([[0, 0, 2], [0, 2, 0], [2, 0, 0]], rotation=[0, 2, 1])
    inverse = dist.inv(grid)
    assert numpy.allclose(dist.fwd(inverse), grid)
    assert numpy.allclose(dist.pdf(inverse),
                          [2.550e-05, 2.553e-05, 2.553e-05, 2.552e-05,
                           2.525e-05, 2.522e-05, 2.522e-05, 2.519e-05])
    assert numpy.allclose(
        inverse,
        [[-1.38424, -1.38424, -1.38424, -1.38424,  3.19971,  3.19971,  3.19971,  3.19971],
         [ 0.51571, -1.48391,  3.48391,  1.48413, -1.48391, -1.48391,  1.48429,  1.48429],
         [-1.31003,  3.31003, -1.31003,  3.31003, -1.48391,  1.48429, -1.48391,  1.48429]],
        rtol=1e-5,
    )
