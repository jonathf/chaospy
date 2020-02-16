import numpy
import chaospy


def test_dependent_pdf():
    distribution1 = chaospy.Exponential(1)
    distribution2 = chaospy.Uniform(lower=0, upper=distribution1)
    distribution = chaospy.J(distribution1, distribution2)
    assert distribution.pdf([0.5, 0.6]) == 0
    assert distribution.pdf([0.5, 0.4]) > 0
