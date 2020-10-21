"""Folded Cauchy distribution."""
import numpy

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class folded_cauchy(SimpleDistribution):
    """Folded Cauchy distribution."""

    def __init__(self, c=0):
        super(folded_cauchy, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        return 1./(numpy.pi*(1+(x-c)**2))+1/(numpy.pi*(1+(x+c)**2))

    def _cdf(self, x, c):
        return (numpy.arctan(x-c)+numpy.arctan(x+c))/numpy.pi

    def _lower(self, c):
        return 0.

    def _upper(self, c):
        return 1e+16  # actually infinity


class FoldedCauchy(ShiftScaleDistribution):
    """
    Folded Cauchy distribution.

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.FoldedCauchy(1.5)
        >>> distribution
        FoldedCauchy(1.5)
        >>> uloc = numpy.linspace(0.1, 0.9, 5)
        >>> uloc
        array([0.1, 0.3, 0.5, 0.7, 0.9])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.489, 1.217, 1.803, 2.67 , 6.644])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.222, 0.333, 0.318, 0.152, 0.016])
        >>> distribution.sample(4).round(3)
        array([1.929, 8.542, 0.311, 1.414])

    Notes:
        The Cauchy distribution is what is known as a "pathological"
        distribution. It is not only infinitely bound, but heavy tailed
        enough that approximate bounds is also infinite for any reasonable
        approximation. This makes both bounds and moments results in
        non-sensibel results. In the case of folded-Cauchy distribution::

            >>> distribution.upper > 1e10
            array([ True])

    """

    def __init__(self, shape=0, scale=1, shift=0):
        super(FoldedCauchy, self).__init__(
            dist=folded_cauchy(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
