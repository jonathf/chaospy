"""Multivariate Student-T Distribution."""
import numpy
from scipy import special
import chaospy

from .student_t import student_t
from ..baseclass import MeanCovarianceDistribution


class MvStudentT(MeanCovarianceDistribution):
    """
    Multivariate Student-T Distribution.

    Args:
        df (float, Distribution):
            Degree of freedom
        mu (numpy.ndarray, Distribution):
            Location parameter
        sigma (numpy.ndarray):
            Covariance matrix.

    Examples:
        >>> distribution = chaospy.MvStudentT(40, [1, 2], [[1, 0.6], [0.6, 1]])
        >>> distribution
        MvStudentT(df=40, mu=[1, 2], sigma=[[1, 0.6], [0.6, 1]])
        >>> chaospy.Cov(distribution).round(4)
        array([[1.0526, 0.6316],
               [0.6316, 1.0526]])
        >>> mesh = numpy.mgrid[0.25:0.75:3j, 0.25:0.75:2j].reshape(2, -1)
        >>> mesh.round(4)
        array([[0.25, 0.25, 0.5 , 0.5 , 0.75, 0.75],
               [0.25, 0.75, 0.25, 0.75, 0.25, 0.75]])
        >>> inverse_map = distribution.inv(mesh)
        >>> inverse_map.round(4)
        array([[0.3193, 0.3193, 1.    , 1.    , 1.6807, 1.6807],
               [1.0471, 2.1361, 1.4555, 2.5445, 1.8639, 2.9529]])
        >>> numpy.allclose(distribution.fwd(inverse_map), mesh)
        True
        >>> distribution.pdf(inverse_map).round(4)
        array([0.1225, 0.1225, 0.1552, 0.1552, 0.1225, 0.1225])
        >>> distribution.sample(4).round(4)
        array([[ 1.3979, -0.2189,  2.6868,  0.9551],
               [ 3.1625,  0.6234,  1.582 ,  1.7631]])

    """

    def __init__(
            self,
            df,
            mu,
            sigma=None,
            rotation=None,
    ):
        super(MvStudentT, self).__init__(
            dist=student_t(df),
            mean=mu,
            covariance=sigma,
            rotation=rotation,
            repr_args=chaospy.format_repr_kwargs(df=(df, None))+
                      chaospy.format_repr_kwargs(mu=(mu, None))+
                      chaospy.format_repr_kwargs(sigma=(sigma, None)),
        )
