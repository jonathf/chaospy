"""
Frontend for the collection distributions.

This modules provides a wrapper with documentation for the dist.cores module.
"""
import logging
from functools import wraps
import numpy
from scipy.stats import gaussian_kde

from . import collection



def OTDistribution(distribution):
    """
    OpenTURNS distribution.

    Args:
        distribution (openturns.Distribution, Dist) : underlying OT distribution
    """
    dist = cores.otdistribution(distribution)
    return dist


def deprecation_warning(func):
    """Add a deprecation warning do each distribution."""
    @wraps(func)
    def caller(*args, **kwargs):
        """Docs to be replaced."""
        logger = logging.getLogger(__name__)
        instance = func(*args, **kwargs)
        logger.warning("""\
Distribution `chaospy.{}` is renamed to `chaospy.{}`. Please adjust your code.\
        """.format(func.__name__, instance.__class__.__name__))
        return instance
    return caller


@deprecation_warning
def Alpha(shape=1, scale=1, shift=0):
    """
    Alpha distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scale Parameter
        shift (float, Dist) : Location of lower threshold
    """
    return collection.Alpha(shape, scale, shift)


@deprecation_warning
def Anglit(loc=0, scale=1):
    """
    Anglit distribution.

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter
    """
    return collection.Anglit(loc, scale)


@deprecation_warning
def Arcsinus(shape=0.5, lo=0, up=1):
    """
    Generalized Arc-sinus distribution

    shape : float, Dist
        Shape parameter where 0.5 is the @deprecation_warning
default non-generalized case.
    lo : float, Dist
        Lower threshold
    up : float, Dist
        Upper threshold
    """
    return collection.ArcSinus(shape, lo, up)


@deprecation_warning
def Beta(a, b, lo=0, up=1):
    R"""
    Beta Probability Distribution.

    Args:
        a (float, Dist) : First shape parameter, a > 0
        b (float, Dist) : Second shape parameter, b > 0
        lo (float, Dist) : Lower threshold
        up (float, Dist) : Upper threshold

    Examples:
        >>> f = chaospy.Beta(2, 2, 2, 3)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(f.inv(q), 4))
        [2.2871 2.4329 2.5671 2.7129]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(f.sample(4), 4))
        [2.6039 2.2112 2.8651 2.4881]
        >>> print(f.mom(1))
        2.5
    """
    return collection.Beta(a, b, lo, up)


@deprecation_warning
def Bradford(shape=1, lo=0, up=1):
    """
    Bradford distribution.

    Args:
        shape (float, Dist) : Shape parameter
        lo (float, Dist) : Location of lower threshold
        up (float, Dist) : Location of upper threshold
    """
    return collection.Bradford(shape, lo, up)


@deprecation_warning
def Burr(c=1, d=1, loc=0, scale=1):
    """
    Burr Type III distribution.

    Args:
        c (float, Dist) : Shape parameter
        d (float, Dist) : Shape parameter
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter
    """
<<<<<<< HEAD
    dist = cores.burr(c=c, d=d)*scale + loc
    dist.addattr(str="Burr(%s,%s,%s,%s)"%(c, d, loc, scale))
    return dist
||||||| merged common ancestors
    dist = cores.burr(c=1., d=1.)*scale + loc
    dist.addattr(str="Burr(%s,%s,%s,%s)"%(c, d, loc, scale))
    return dist
=======
    return collection.Burr(c, d, loc, scale)
>>>>>>> replace graph backend: step 1


@deprecation_warning
def Cauchy(loc=0, scale=1):
    """
    Cauchy distribution.

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter
    """
    return collection.Cauchy(loc, scale)


@deprecation_warning
def Chi(df=1, scale=1, shift=0):
    """
    Chi distribution.

    Args:
        df (float, Dist) : Degrees of freedom
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.Chi(df, scale, shift)


@deprecation_warning
def Chisquard(df=1, scale=1, shift=0, nc=0):
    """
    (Non-central) Chi-squared distribution.

    Args:
        df (float, Dist) : Degrees of freedom
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
        nc (float, Dist) : Non-centrality parameter
    """
    return collection.ChiSquard(df, scale, shift, nc)


@deprecation_warning
def Dbl_gamma(shape=1, scale=1, shift=0):
    """
    Double gamma distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.DoubleGamma(shape, scale, shift)


@deprecation_warning
def Dbl_weibull(shape=1, scale=1, shift=0):
    """
    Double weibull distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.DoubleWeibull(shape, scale, shift)


@deprecation_warning
def Exponential(scale=1, shift=0):
    R"""
    Exponential Probability Distribution

    Args:
        scale (float, Dist) : Scale parameter. scale!=0
        shift (float, Dist) : Location of the lower bound.

    Examples;:
        >>> f = chaospy.Exponential(1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(f.inv(q), 4))
        [0.2231 0.5108 0.9163 1.6094]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(f.sample(4), 4))
        [1.0601 0.1222 3.0014 0.6581]
        >>> print(f.mom(1))
        1.0
    """
    return collection.Exponential(scale, shift)


@deprecation_warning
def Exponpow(shape=0, scale=1, shift=0):
    """
    Expontial power distribution.

    Also known as Generalized error distribution and Generalized normal
    distribution version 1.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.ExponentialPower(shape, scale, shift)


@deprecation_warning
def Exponweibull(a=1, c=1, scale=1, shift=0):
    """
    Expontiated Weibull distribution.

    Args:
        a (float, Dist) : First shape parameter
        c (float, Dist) : Second shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.ExponentialWeibull(a, c, scale, shift)


@deprecation_warning
def F(n=1, m=1, scale=1, shift=0, nc=0):
    """
    (Non-central) F or Fisher-Snedecor distribution.

    Args:
        n (float, Dist) : Degres of freedom for numerator
        m (float, Dist) : Degres of freedom for denominator
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
        nc (float, Dist) : Non-centrality parameter
    """
    return collection.F(n, m, scale, shift, nc)


@deprecation_warning
def Fatiguelife(shape=1, scale=1, shift=0):
    """
    Fatigue-Life or Birmbaum-Sanders distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.FatigueLife(shape, scale, shift)


@deprecation_warning
def Fisk(shape=1, scale=1, shift=0):
    """
    Fisk or Log-logistic distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.Fisk(shape, scale, shift)


@deprecation_warning
def Foldcauchy(shape=0, scale=1, shift=0):
    """
    Folded Cauchy distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.FoldCauchy(shape, scale, shift)


@deprecation_warning
def Foldnormal(mu=0, sigma=1, loc=0):
    """
    Folded normal distribution.

    Args:
        mu (float, Dist) : Location parameter in normal distribution
        sigma (float, Dist) : Scaling parameter (in both normal and fold)
        loc (float, Dist) : Location of fold
    """
    return collection.FoldNormal(mu, sigma, loc)


@deprecation_warning
def Frechet(shape=1, scale=1, shift=0):
    """
    Frechet or Extreme value distribution type 2.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.Frechet(shape, scale, shift)


@deprecation_warning
def Gamma(shape=1, scale=1, shift=0):
    """
    Gamma distribution.

    Also an Erlang distribution when shape=k and scale=1./lamb.

    Args:
        shape (float, Dist) : Shape parameter. a>0
        scale () : Scale parameter. scale!=0
        shift (float, Dist) : Location of the lower bound.

    Examples:
        >>> f = chaospy.Gamma(1, 1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(f.inv(q), 4))
        [0.2231 0.5108 0.9163 1.6094]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(f.sample(4), 4))
        [1.0601 0.1222 3.0014 0.6581]
        >>> print(f.mom(1))
        1.0
    """
    return collection.Gamma(shape, scale, shift)


@deprecation_warning
def Genexpon(a=1, b=1, c=1, scale=1, shift=0):
    """
    Generalized exponential distribution.

    Args:
        a (float, Dist) : First shape parameter
        b (float, Dist) : Second shape parameter
        c (float, Dist) : Third shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter

    Note:
        "An Extension of Marshall and Olkin's Bivariate Exponential Distribution",
        H.K. Ryu, Journal of the American Statistical Association, 1993.

        "The Exponential Distribution: Theory, Methods and Applications",
        N. Balakrishnan, Asit P. Basu.
    """
    return collection.GeneralizedExponential(a, b, c, scale, shift)


@deprecation_warning
def Genextreme(shape=0, scale=1, loc=0):
    """
    Generalized extreme value distribution
    Fisher-Tippett distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        loc (float, Dist) : Location parameter
    """
    return collection.GeneralizedExtreme(shape, scale, loc)


@deprecation_warning
def Gengamma(shape1, shape2, scale, shift):
    """
    Generalized gamma distribution

    Args:
        shape1 (float, Dist) : Shape parameter 1
        shape2 (float, Dist) : Shape parameter 2
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.GeneralizedGamma(shape1, shape2, scale, shift)


@deprecation_warning
def Genhalflogistic(shape, scale, shift):
    """
    Generalized half-logistic distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.GeneralizedHalfLogistic(shape, scale, shift)


@deprecation_warning
def Gilbrat(scale=1, shift=0):
    """
    Gilbrat distribution.

    Standard log-normal distribution

    Args:
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.Gilbrat(scale, shift)


@deprecation_warning
def Gompertz(shape, scale, shift):
    """
    Gompertz distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.Gompertz(shape, scale, shift)


@deprecation_warning
def Logweibul(scale=1, loc=0):
    """
    Gumbel or Log-Weibull distribution.

    Args:
        scale (float, Dist) : Scaling parameter
        loc (float, Dist) : Location parameter
    """
    return collection.LogWeibull(scale, loc)


@deprecation_warning
def Hypgeosec(loc=0, scale=1):
    """
    hyperbolic secant distribution

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scale parameter
    """
    return collection.HyperbolicSecant(loc, scale)


@deprecation_warning
def Kumaraswamy(a, b, lo=0, up=1):
    """
    Kumaraswswamy's double bounded distribution

    Args:
        a (float, Dist) : First shape parameter
        b (float, Dist) : Second shape parameter
        lo (float, Dist) : Lower threshold
        up (float, Dist) : Upper threshold
    """
    return collection.Kumaraswamy(a, b, lo, up)


@deprecation_warning
def Laplace(mu=0, scale=1):
    R"""
    Laplace Probability Distribution

    Args:
        mu (float, Dist) : Mean of the distribution.
        scale (float, Dist) : Scaleing parameter.
            scale > 0

    Examples:
        >>> f = chaospy.Laplace(2, 2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(f.inv(q), 4))
        [0.1674 1.5537 2.4463 3.8326]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(f.sample(4), 4))
        [ 2.734  -0.9392  6.6165  1.9275]
        >>> print(f.mom(1))
        2.0
    """
    return collection.Laplace(mu, scale)


@deprecation_warning
def Levy(loc=0, scale=1):
    """
    Levy distribution

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter
    """
    return collection.Levy(loc, scale)


@deprecation_warning
def Loggamma(shape=1, scale=1, shift=0):
    """
    Log-gamma distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.LogGamma(shape, scale, shift)


@deprecation_warning
def Logistic(loc=0, scale=1, skew=1):
    """
    Generalized logistic type 1 distribution
    Sech squared distribution

    loc (float, Dist) : Location parameter
    scale (float, Dist) : Scale parameter
    skew (float, Dist) : Shape parameter
    """
    return collection.Logistic(loc, scale, skew)


@deprecation_warning
def Loglaplace(shape=1, scale=1, shift=0):
    """
    Log-laplace distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.LogLaplace(shape, scale, shift)


@deprecation_warning
def Lognormal(mu=0, sigma=1, shift=0, scale=1):
    R"""
    Log-normal distribution

    Args:
        mu (float, Dist) : Mean in the normal distribution.  Overlaps with
                scale by mu=log(scale)
        sigma (float, Dist) : Standard deviation of the normal distribution.
        shift (float, Dist) : Location of the lower bound.
        scale (float, Dist) : Scale parameter. Overlaps with mu by scale**mu

    Examples:
        >>> f = chaospy.Lognormal(0, 1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(f.inv(q), 4))
        [0.431  0.7762 1.2883 2.3201]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(f.sample(4), 4))
        [1.4844 0.3011 5.1945 0.9563]
        >>> print(numpy.around(f.mom(1), 4))
        1.6487
    """
    return collection.LogNormal(mu, sigma, shift, scale)


@deprecation_warning
def Loguniform(lo=0, up=1, scale=1, shift=0):
    """
    Log-uniform distribution

    Args:
        lo (float, Dist) : Location of lower threshold
        up (float, Dist) : Location of upper threshold
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.LogUniform(lo, up, scale, shift)


@deprecation_warning
def Maxwell(scale=1, shift=0):
    """
    Maxwell-Boltzmann distribution
    Chi distribution with 3 degrees of freedom

    Args:
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.Maxwell(scale, shift)


@deprecation_warning
def Mielke(kappa=1, expo=1, scale=1, shift=0):
    """
    Mielke's beta-kappa distribution

    Args:
        kappa (float, Dist) : First shape parameter
        expo (float, Dist) : Second shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.Mielke(kappa, expo, scale, shift)


@deprecation_warning
def MvLognormal(loc=[0,0], scale=[[1,.5],[.5,1]]):
    """
    Multivariate Log-Normal Distribution.

    Args:
        loc (float, Dist) : Mean vector
        scale (float, Dist) : Covariance matrix or variance vector if scale is
                a 1-d vector.
    """
    return collection.MvLognormal(loc, scale)


@deprecation_warning
def MvNormal(loc=[0,0], scale=[[1,.5],[.5,1]]):
    """
    Multivariate Normal Distribution

    Args:
        loc (float, Dist) : Mean vector
        scale (float, Dist) : Covariance matrix or variance vector if scale is a 1-d vector.

    Examples:
        >>> f = chaospy.MvNormal([0,0], [[1,.5],[.5,1]])
        >>> q = [[.4,.5,.6],[.4,.5,.6]]
        >>> print(numpy.around(f.inv(q), 4))
        [[-0.2533  0.      0.2533]
         [-0.3461  0.      0.3461]]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [[0.4 0.5 0.6]
         [0.4 0.5 0.6]]
        >>> print(numpy.around(f.sample(3), 4))
        [[ 0.395  -1.2003  1.6476]
         [ 0.1588  0.3855  0.1324]]
        >>> print(numpy.around(f.mom((1,1)), 4))
        0.5
    """
    return collection.MvNormal(loc, scale)


@deprecation_warning
def MvStudent_t(df=1, loc=[0,0], scale=[[1,.5],[.5,1]]):
    """
    Args:
        df (float, Dist) : Degree of freedom
        loc (array_like, Dist) : Location parameter
        scale (array_like) : Covariance matrix
    """
    return collection.MvStudentT(df, loc, scale)


@deprecation_warning
def Nakagami(shape=1, scale=1, shift=0):
    """
    Nakagami-m distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.Nakagami(shape, scale, shift)


@deprecation_warning
def Pareto1(shape=1, scale=1, loc=0):
    """
    Pareto type 1 distribution.

    Lower threshold at scale+loc and survival: x^-shape

    Args:
        shape (float, Dist) : Tail index parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.Pareto1(shape, scale, loc)


@deprecation_warning
def Pareto2(shape=1, scale=1, loc=0):
    """
    Pareto type 2 distribution.

    Also known as Lomax distribution (for loc=0).

    Lower threshold at loc and survival: (1+x)^-shape.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        loc (float, Dist) : Location parameter
    """
    return collection.Pareto2(shape, scale, loc)


@deprecation_warning
def Powerlaw(shape=1, lo=0, up=1):
    """
    Powerlaw distribution

    Args:
        shape (float, Dist) : Shape parameter
        lo (float, Dist) : Location of lower threshold
        up (float, Dist) : Location of upper threshold
    """
    return collection.PowerLaw(shape, lo, up)


@deprecation_warning
def Powerlognormal(shape=1, mu=0, sigma=1, shift=0, scale=1):
    """
    Power log-normal distribution

    Args:
        shape (float, Dist) : Shape parameter
        mu (float, Dist) : Mean in the normal distribution.  Overlaps with
                scale by mu=log(scale)
        sigma (float, Dist) : Standard deviation of the normal distribution.
        shift (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter. Overlap with mu in scale**mu
    """
    return collection.PowerLogNormal(shape, mu, sigma, shift, scale)


@deprecation_warning
def Powernorm(shape=1, mu=0, scale=1):
    """
    Power normal or Box-Cox distribution.

    Args:
        shape (float, Dist) : Shape parameter
        mu (float, Dist) : Mean of the normal distribution
        scale (float, Dist) : Standard deviation of the normal distribution
    """
    return collection.PowerNormal(shape, mu, scale)


@deprecation_warning
def Raised_cosine(loc=0, scale=1):
    """
    Raised cosine distribution

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scale parameter
    """
    return collection.RaisedCosine(loc, scale)


@deprecation_warning
def Rayleigh(scale=1, shift=0):
    """
    Rayleigh distribution

    Args:
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.Rayleigh(scale, shift)


@deprecation_warning
def Reciprocal(lo=1, up=2):
    """
    Reciprocal distribution

    Args:
        lo (float, Dist) : Location of lower threshold
        up (float, Dist) : Location of upper threshold
    """
    return collection.Reciprocal(lo, up)


@deprecation_warning
def Student_t(df, loc=0, scale=1, nc=0):
    """
    (Non-central) Student-t distribution

    Args:
        df (float, Dist) : Degrees of freedom
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scale parameter
        nc (flat, Dist) : Non-centrality parameter
    """
    return collection.StudentT(df, loc, scale, nc)


@deprecation_warning
def Triangle(lo, mid, up):
    """
    Triangle Distribution.

    Must have lo <= mid <= up.

    Args:
        lo (float, Dist) : Lower bound
        mid (float, Dist) : Location of the top
        up (float, Dist) : Upper bound

    Examples:
        >>> f = chaospy.Triangle(2, 3, 4)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(f.inv(q), 4))
        [2.6325 2.8944 3.1056 3.3675]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(f.sample(4), 4))
        [3.1676 2.4796 3.6847 2.982 ]
        >>> print(numpy.around(f.mom(1), 4))
        3.0
    """
    return collection.Triangle(lo, mid, up)


@deprecation_warning
def Truncexpon(up=1, scale=1, shift=0):
    """
    Truncated exponential distribution.

    Args:
        up (float, Dist) : Location of upper threshold
        scale (float, Dist) : Scaling parameter in the exponential distribution
        shift (float, Dist) : Location parameter
    """
    return collection.TruncExponential(up, scale, shift)


@deprecation_warning
def Truncnorm(lo=-1, up=1, mu=0, sigma=1):
    """
    Truncated normal distribution

    Args:
        lo (float, Dist) : Location of lower threshold
        up (float, Dist) : Location of upper threshold
        mu (float, Dist) : Mean of normal distribution
        sigma (float, Dist) : Standard deviation of normal distribution
    """
    return collection.TruncNormal(lo, up, mu, sigma)


@deprecation_warning
def Tukeylambda(shape=0, scale=1, shift=0):
    """
    Tukey-lambda distribution

    Args:
        lam (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.TukeyLambda(shape, scale, shift)


@deprecation_warning
def Wald(mu=0, scale=1, shift=0):
    """
    Wald distribution
    Reciprocal inverse Gaussian distribution

    Args:
        mu (float, Dist) : Mean of the normal distribution
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.Wald(mu, scale, shift)


@deprecation_warning
def Weibull(shape=1, scale=1, shift=0):
    """
    Weibull Distribution

    Args:
        shape (float, Dist) : Shape parameter.
        scale (float, Dist) : Scale parameter.
        shift (float, Dist) : Location of lower bound.

    Examples:
        >>> f = chaospy.Weibull(2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(f.inv(q), 4))
        [0.4724 0.7147 0.9572 1.2686]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(f.sample(4), 4))
        [1.0296 0.3495 1.7325 0.8113]
        >>> print(numpy.around(f.mom(1), 4))
        0.8862
    """
    return collection.Weibull(shape, scale, shift)


@deprecation_warning
def Wigner(radius=1, shift=0):
    """
    Wigner (semi-circle) distribution

    Args:
        radius (float, Dist) : radius of the semi-circle (scale)
        shift (float, Dist) : location of the origen (location)
    """
    return collection.Wigner(radius, shift)


@deprecation_warning
def Wrapcauchy(shape=0.5, scale=1, shift=0):
    """
    Wrapped Cauchy distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    return collection.WrappedCauchy(shape, scale, shift)


@deprecation_warning
def SampleDist(samples, lo=None, up=None):
    """
    Distribution based on samples.

    Estimates a distribution from the given samples by constructing a kernel
    density estimator (KDE).

    Args:
        samples:
            Sample values to construction of the KDE
        lo (float) : Location of lower threshold
        up (float) : Location of upper threshold
    """
    return collection.SampleDist(samples, lo, up)
