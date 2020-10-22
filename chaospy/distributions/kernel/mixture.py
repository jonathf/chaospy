"""Gaussian Mixture Model."""
import numpy

from .gaussian import GaussianKDE


class GaussianMixture(GaussianKDE):
    """
    Gaussian Mixture Model.

    A Gaussian mixture model is a probabilistic model that assumes all the data
    points are generated from a mixture of a finite number of Gaussian
    distributions with unknown parameters. One can think of mixture models as
    generalizing K-means clustering to incorporate information about the
    covariance structure of the data as well as the centers of the latent
    Gaussians.

    Attributes:
        means:
            Sequence of means.
        covariances:
            Sequence of covariance matrices.
        weights:
            How much each sample is weighted. Either a scalar when the samples
            are equally weighted, or a vector with the same length as the
            number of mixed models.

    Examples:
        >>> means = ([0, 1], [1, 0])
        >>> covariances = ([[1, 0], [0, 1]], [[1, 0.5], [0.5, 1]])
        >>> distribution = GaussianMixture(means, covariances)
        >>> uloc = [[0, 0, 1, 1], [0, 1, 0, 1]]
        >>> distribution.pdf(uloc).round(4)
        array([0.0954, 0.092 , 0.1212, 0.0954])
        >>> distribution.fwd(uloc).round(4)
        array([[0.3293, 0.3293, 0.6707, 0.6707],
               [0.3699, 0.6731, 0.3711, 0.734 ]])
        >>> distribution.inv(uloc).round(4)
        array([[-8.9681, -8.9681,  8.0521,  8.0521],
               [-9.862 , 10.1977, -9.5929, 10.2982]])
        >>> distribution.mom([(0, 1, 1), (1, 0, 1)]).round(4)
        array([0.5 , 0.5 , 0.25])

    """

    @property
    def means(self):
        return self._samples.T

    @property
    def covariances(self):
        return numpy.swapaxes(self._covariance, 0, 2)

    def __init__(self, means, covariances, weights=None, rotation=None):
        """
        Args:
            means (numpy.ndarray):
                Sequence of mean values. With shape `(n_components, n_dim)`.
            covariances (numpy.ndarray):
                Sequence of covariance matrices.
                With shape `(n_components, n_dim, n_dim)`.
            weights (Optional[numpy.ndarray]):
                Weights of the samples. This must have the shape
                `(n_components,)`. If omitted, each sample is assumed to be
                equally weighted.

        """
        means = numpy.atleast_2d(numpy.transpose(means))
        n, m = means.shape

        covariances = numpy.asfarray(covariances)
        if covariances.ndim in (1, 2):
            covariances = numpy.broadcast_to(covariances.T, (n, n, m))
        else:
            covariances = numpy.swapaxes(covariances, 0, 2)
        assert covariances.shape == (n, n, m)
        super(GaussianMixture, self).__init__(
            samples=means, h_mat=covariances, weights=weights, rotation=rotation)
