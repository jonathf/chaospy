"""Wrapper to the OpenTURNS distribution."""
import numpy

from ..baseclass import Dist


class OTDistribution(Dist):
    """OpenTURNS distribution."""

    def __init__(self, distribution):
        """
        Args:
            distribution (openturns.Distribution):
                1D distribution created in OpenTURNS.
        """
        Dist.__init__(self)
        if distribution.getDimension() != 1:
            raise Exception("Only 1D OpenTURNS distribution are supported for now")
        self.distribution = distribution

    def _pdf(self, x):
        return numpy.array(self.distribution.computePDF(numpy.atleast_2d(x).T).asPoint())

    def _cdf(self, x):
        return numpy.array(self.distribution.computeCDF(numpy.atleast_2d(x).T).asPoint())

    def _ppf(self, q):
        return numpy.array(self.distribution.computeQuantile(q[0]).asPoint())

    def _bnd(self):
        rng = self.distribution.getRange()
        return rng.getLowerBound()[0], rng.getUpperBound()[0]

    def _mom(self, k):
        return self.getMoment(k)[0]

    def _str(self):
        return self.distribution.__str__()
