"""
Frontend for the collection distributions.

This modules provides a wrapper with documentation for the dist.cores module.
"""
import numpy
from scipy.stats import gaussian_kde

from . import cores, joint


def Alpha(shape=1, scale=1, shift=0):
    """
    Alpha distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scale Parameter
        shift (float, Dist) : Location of lower threshold
    """
    dist = cores.alpha(shape)*scale + shift
    dist.addattr(str="Alpha(%s,%s,%s)" % (shape, scale, shift))
    return dist


def Anglit(loc=0, scale=1):
    """
    Anglit distribution.

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter
    """
    dist = cores.anglit()*scale + loc
    dist.addattr(str="Anglit(%s,%s)"(loc, scale))
    return dist


def Arcsinus(shape=0.5, lo=0, up=1):
    """
    Generalized Arc-sinus distribution

    shape : float, Dist
        Shape parameter where 0.5 is the default non-generalized case.
    lo : float, Dist
        Lower threshold
    up : float, Dist
        Upper threshold
    """
    dist = cores.beta(shape, 1-shape)*(up-lo) + lo
    dist.addattr(str="Arcsinus(%s,%s,%s)" % (shape, lo, up))
    return dist


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
    dist = cores.beta(a, b)*(up-lo) + lo
    dist.addattr(str="Beta(%s,%s,%s,%s)" % (a,b,lo,up))
    return dist


def Bradford(shape=1, lo=0, up=1):
    """
    Bradford distribution.

    Args:
        shape (float, Dist) : Shape parameter
        lo (float, Dist) : Location of lower threshold
        up (float, Dist) : Location of upper threshold
    """
    dist = cores.bradford(c=shape)*(up-lo) + lo
    dist.addattr(str="Bradford(%s,%s,%s)"%(shape, lo, up))
    return dist


def Burr(c=1, d=1, loc=0, scale=1):
    """
    Burr Type XII or Singh-Maddala distribution.

    Args:
        c (float, Dist) : Shape parameter
        d (float, Dist) : Shape parameter
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter
    """
    dist = cores.burr(c=1., d=1.)*scale + loc
    dist.addattr(str="Burr(%s,%s,%s,%s)"%(c, d, loc, scale))
    return dist


def Cauchy(loc=0, scale=1):
    """
    Cauchy distribution.

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter
    """
    dist = cores.cauchy()*scale + loc
    dist.addattr(str="Cauchy(%s,%s)"%(loc,scale))
    return dist


def Chi(df=1, scale=1, shift=0):
    """
    Chi distribution.

    Args:
        df (float, Dist) : Degrees of freedom
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.chi(df)*scale + shift
    dist.addattr(str="Chi(%s,%s,%s)"%(df, scale, shift))
    return dist


def Chisquard(df=1, scale=1, shift=0, nc=0):
    """
    (Non-central) Chi-squared distribution.

    Args:
        df (float, Dist) : Degrees of freedom
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
        nc (float, Dist) : Non-centrality parameter
    """
    dist = cores.chisquared(df, nc)*scale + shift
    dist.addattr(str="Chisquared(%s,%s,%s,%s)"%(df, nc,scale,shift))
    return dist


def Dbl_gamma(shape=1, scale=1, shift=0):
    """
    Double gamma distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.dbl_gamma(shape)*scale + shift
    dist.addattr(str="Dbl_gamma(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Dbl_weibull(shape=1, scale=1, shift=0):
    """
    Double weibull distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.dbl_weibull(shape)*scale + shift
    dist.addattr(str="Dbl_weibull(%s,%s,%s)"%(shape, scale, shift))
    return dist


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
    dist = cores.expon()*scale + shift
    dist.addattr(str="Expon(%s,%s)" % (scale, shift))
    return dist


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
    dist = cores.exponpow(shape)*scale + shift
    dist.addattr(str="Exponpow(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Exponweibull(a=1, c=1, scale=1, shift=0):
    """
    Expontiated Weibull distribution.

    Args:
        a (float, Dist) : First shape parameter
        c (float, Dist) : Second shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.exponweibull(a, c)*scale + shift
    dist.addattr(str="Exponweibull(%s,%s,%s,%s)"%(a, c, scale,shift))
    return dist


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
    dist = cores.f(n, m, nc)*scale + shift
    dist.addattr(str="F(%s,%s,%s,%s,%s)"%(n, m, scale, shift, nc))
    return dist


def Fatiguelife(shape=1, scale=1, shift=0):
    """
    Fatigue-Life or Birmbaum-Sanders distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.fatiguelife(shape)*scale + shift
    dist.addattr(str="Fatiguelife(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Fisk(shape=1, scale=1, shift=0):
    """
    Fisk or Log-logistic distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.fisk(c=shape)*scale + shift
    dist.addattr(str="Fisk(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Foldcauchy(shape=0, scale=1, shift=0):
    """
    Folded Cauchy distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.foldcauchy(shape)*scale + shift
    dist.addattr(str="Foldcauchy(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Foldnormal(mu=0, sigma=1, loc=0):
    """
    Folded normal distribution.

    Args:
        mu (float, Dist) : Location parameter in normal distribution
        sigma (float, Dist) : Scaling parameter (in both normal and fold)
        loc (float, Dist) : Location of fold
    """
    dist = cores.foldnorm(mu-loc)*sigma + loc
    dist.addattr(str="Foldnorm(%s,%s,%s)"%(mu, sigma, loc))
    return dist


def Frechet(shape=1, scale=1, shift=0):
    """
    Frechet or Extreme value distribution type 2.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.frechet(shape)*scale + shift
    dist.addattr(str="Frechet(%s,%s,%s)"%(shape, scale, shift))
    return dist


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
    dist = cores.gamma(shape)*scale + shift
    dist.addattr(str="Gamma(%s,%s,%s)"%(shape, scale, shift))
    return dist


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
    dist = cores.genexpon(a=1, b=1, c=1)*scale + shift
    dist.addattr(str="Genexpon(%s,%s,%s)"%(a, b, c))
    return dist


def Genextreme(shape=0, scale=1, loc=0):
    """
    Generalized extreme value distribution
    Fisher-Tippett distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        loc (float, Dist) : Location parameter
    """
    dist = cores.genextreme(shape)*scale + loc
    dist.addattr(str="Genextreme(%s,%s,%s)"%(shape, scale, loc))
    return dist


def Gengamma(shape1, shape2, scale, shift):
    """
    Generalized gamma distribution

    Args:
        shape1 (float, Dist) : Shape parameter 1
        shape2 (float, Dist) : Shape parameter 2
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.gengamma(shape1, shape2)*scale + shift
    dist.addattr(
        str="Gengamma(%s,%s,%s,%s)"%(shape1,shape2,scale,shift))
    return dist


def Genhalflogistic(shape, scale, shift):
    """
    Generalized half-logistic distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.genhalflogistic(shape)*scale + shift
    dist.addattr(str="Genhalflogistic(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Gilbrat(scale=1, shift=0):
    """
    Gilbrat distribution.

    Standard log-normal distribution

    Args:
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.lognormal(1)*scale + shift
    dist.addattr(str="Gilbrat(%s,%s)"%(scale, shift))
    return dist


def Gompertz(shape, scale, shift):
    """
    Gompertz distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.gompertz(shape)*scale + shift
    dist.addattr(str="Gompertz(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Logweibul(scale=1, loc=0):
    """
    Gumbel or Log-Weibull distribution.

    Args:
        scale (float, Dist) : Scaling parameter
        loc (float, Dist) : Location parameter
    """
    dist = cores.gumbel()*scale + loc
    dist.addattr(str="Gumbel(%s,%s)"%(scale, loc))
    return dist


def Hypgeosec(loc=0, scale=1):
    """
    hyperbolic secant distribution

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scale parameter
    """
    dist = cores.hypgeosec()*scale + loc
    dist.addattr(str="Hypgeosec(%s,%s)"%(loc, scale))
    return dist


def Kumaraswamy(a, b, lo=0, up=1):
    """
    Kumaraswswamy's double bounded distribution

    Args:
        a (float, Dist) : First shape parameter
        b (float, Dist) : Second shape parameter
        lo (float, Dist) : Lower threshold
        up (float, Dist) : Upper threshold
    """
    dist = cores.kumaraswamy(a,b)*(up-lo) + lo
    dist.addattr(str="Kumaraswamy(%s,%s,%s,%s)"%(a,b,lo,up))
    return dist


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
    dist = cores.laplace()*scale + mu
    dist.addattr(str="Laplace(%s,%s)"%(mu,scale))
    return dist


def Levy(loc=0, scale=1):
    """
    Levy distribution

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter
    """
    dist = cores.levy()*scale+loc
    dist.addattr(str="Levy(%s,%s)"%(loc, scale))
    return dist


def Loggamma(shape=1, scale=1, shift=0):
    """
    Log-gamma distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.loggamma(shape)*scale + shift
    dist.addattr(str="Loggamma(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Logistic(loc=0, scale=1, skew=1):
    """
    Generalized logistic type 1 distribution
    Sech squared distribution

    loc (float, Dist) : Location parameter
    scale (float, Dist) : Scale parameter
    skew (float, Dist) : Shape parameter
    """
    dist = cores.logistic()*scale + loc
    dist.addattr(str="Logistic(%s,%s)"%(loc, scale))
    return dist


def Loglaplace(shape=1, scale=1, shift=0):
    """
    Log-laplace distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.loglaplace(shape)*scale + shift
    dist.addattr(str="Loglaplace(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Lognormal(mu=0, sigma=1, shift=0, scale=1):
    R"""
    Log-normal distribution

    Args:
        mu (float, Dist) : Mean in the normal distribution.  Overlaps with
                scale by mu=log(scale)
        sigma (float, Dist) : Standard deviation of the normal distribution.
        shift (float, Dist) : Location of the lower bound.
        scale (float, Dist) : Scale parameter. Overlaps with mu by scale=e**mu

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
    dist = cores.lognormal(sigma)*scale*numpy.e**mu + shift
    dist.addattr(str="Lognormal(%s,%s,%s,%s)"%(mu,sigma,shift,scale))
    return dist


def Loguniform(lo=0, up=1, scale=1, shift=0):
    """
    Log-uniform distribution

    Args:
        lo (float, Dist) : Location of lower threshold
        up (float, Dist) : Location of upper threshold
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.loguniform(lo, up)*scale + shift
    dist.addattr(str="Loguniform(%s,%s,%s,%s)" % (lo,up,scale,shift))
    return dist


def Maxwell(scale=1, shift=0):
    """
    Maxwell-Boltzmann distribution
    Chi distribution with 3 degrees of freedom

    Args:
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.chi(3)*scale + shift
    dist.addattr(str="Maxwell(%s,%s)"%(scale, shift))
    return dist


def Mielke(kappa=1, expo=1, scale=1, shift=0):
    """
    Mielke's beta-kappa distribution

    Args:
        kappa (float, Dist) : First shape parameter
        expo (float, Dist) : Second shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.mielke(kappa, expo)*scale + shift
    dist.addattr(str="Mielke(%s,%s,%s,%s)"%(kappa,expo,scale,shift))
    return dist


def MvLognormal(loc=[0,0], scale=[[1,.5],[.5,1]]):
    """
    Multivariate Log-Normal Distribution.

    Args:
        loc (float, Dist) : Mean vector
        scale (float, Dist) : Covariance matrix or variance vector if scale is
                a 1-d vector.
    """
    dist = cores.mvlognormal(loc, scale)
    dist.addattr(str="MvLognormal(%s,%s)" % (loc, scale))
    return dist


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
    if numpy.all((numpy.diag(numpy.diag(scale))-scale)==0):
        out = joint.J(
            *[Normal(loc[i], scale[i,i]) for i in range(len(scale))])
    else:
        out = cores.mvnormal(loc, scale)
    out.addattr(str="MvNormal(%s,%s)" % (loc, scale))
    return out


def MvStudent_t(df=1, loc=[0,0], scale=[[1,.5],[.5,1]]):
    """
    Args:
        df (float, Dist) : Degree of freedom
        loc (array_like, Dist) : Location parameter
        scale (array_like) : Covariance matrix
    """
    out = cores.mvstudentt(df, loc, scale)
    out.addattr(str="MvStudent_t(%s,%s,%s)" % (df, loc, scale))
    return out


def Nakagami(shape=1, scale=1, shift=0):
    """
    Nakagami-m distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.nakagami(shape)*scale + shift
    dist.addattr(str="Nakagami(%s,%s,%s)"%(shape,scale,shift))
    return dist


def Normal(mu=0, sigma=1):
    R"""
    Normal (Gaussian) distribution

    Args:
        mu (float, Dist) : Mean of the distribution.
        sigma (float, Dist) : Standard deviation.  sigma > 0

    Examples:
        >>> f = chaospy.Normal(2, 2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(f.inv(q), 4))
        [0.3168 1.4933 2.5067 3.6832]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(f.sample(4), 4))
        [ 2.7901 -0.4006  5.2952  1.9107]
        >>> print(f.mom(1))
        2.0
    """
    dist = cores.normal()*sigma + mu
    dist.addattr(str="Normal(%s,%s)"%(mu, sigma))
    return dist


def Pareto1(shape=1, scale=1, loc=0):
    """
    Pareto type 1 distribution.

    Lower threshold at scale+loc and survival: x^-shape

    Args:
        shape (float, Dist) : Tail index parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.pareto1(shape)*scale + loc
    dist.addattr(str="Pareto(%s,%s,%s)" % (shape, scale, loc))
    return dist


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
    dist = cores.pareto(shape)*scale + loc
    dist.addattr(str="Pareto(%s,%s,%s)"%(shape, scale, loc))
    return dist


def Powerlaw(shape=1, lo=0, up=1):
    """
    Powerlaw distribution

    Args:
        shape (float, Dist) : Shape parameter
        lo (float, Dist) : Location of lower threshold
        up (float, Dist) : Location of upper threshold
    """
    dist = cores.beta(shape, 1)*(up-lo) + lo
    dist.addattr(str="Powerlaw(%s,%s,%s)"%(shape, lo, up))
    return dist


def Powerlognormal(shape=1, mu=0, sigma=1, shift=0, scale=1):
    """
    Power log-normal distribution

    Args:
        shape (float, Dist) : Shape parameter
        mu (float, Dist) : Mean in the normal distribution.  Overlaps with
                scale by mu=log(scale)
        sigma (float, Dist) : Standard deviation of the normal distribution.
        shift (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter. Overlap with mu in scale=e**mu
    """
    dist = cores.powerlognorm(shape, sigma)*scale*numpy.e**mu + shift
    dist.addattr(str="Powerlognorm(%s,%s,%s,%s,%s)"%\
            (shape, mu, sigma, shift, scale))
    return dist


def Powernorm(shape=1, mu=0, scale=1):
    """
    Power normal or Box-Cox distribution.

    Args:
        shape (float, Dist) : Shape parameter
        mu (float, Dist) : Mean of the normal distribution
        scale (float, Dist) : Standard deviation of the normal distribution
    """
    dist = cores.powernorm(shape)*scale + mu
    dist.addattr(str="Powernorm(%s,%s,%s)"%(shape, mu, scale))
    return dist


def Raised_cosine(loc=0, scale=1):
    """
    Raised cosine distribution

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scale parameter
    """
    dist = cores.raised_cosine()*scale + loc
    dist.addattr(str="Raised_cosine(%s,%s)"%(loc,scale))
    return dist


def Rayleigh(scale=1, shift=0):
    """
    Rayleigh distribution

    Args:
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.chi(2)*scale + shift
    dist.addattr(str="Rayleigh(%s,%s)"%(scale, shift))
    return dist


def Reciprocal(lo=1, up=2):
    """
    Reciprocal distribution

    Args:
        lo (float, Dist) : Location of lower threshold
        up (float, Dist) : Location of upper threshold
    """
    dist = cores.reciprocal(lo,up)
    dist.addattr(str="Reciprocal(%s,%s)"%(lo,up))
    return dist


def Student_t(df, loc=0, scale=1, nc=0):
    """
    (Non-central) Student-t distribution

    Args:
        df (float, Dist) : Degrees of freedom
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scale parameter
        nc (flat, Dist) : Non-centrality parameter
    """
    dist = cores.student_t(df)*scale + loc
    dist.addattr(str="Student_t(%s,%s,%s)" % (df, loc, scale))
    return dist


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
    dist = cores.triangle((mid-lo)*1./(up-lo))*(up-lo) + lo
    dist.addattr(str="Triangle(%s,%s,%s)" % (lo, mid, up))
    return dist


def Truncexpon(up=1, scale=1, shift=0):
    """
    Truncated exponential distribution.

    Args:
        up (float, Dist) : Location of upper threshold
        scale (float, Dist) : Scaling parameter in the exponential distribution
        shift (float, Dist) : Location parameter
    """
    dist = cores.truncexpon((up-shift)/scale)*scale + shift
    dist.addattr(str="Truncexpon(%s,%s,%s)"%(up, scale, shift))
    return dist


def Truncnorm(lo=-1, up=1, mu=0, sigma=1):
    """
    Truncated normal distribution

    Args:
        lo (float, Dist) : Location of lower threshold
        up (float, Dist) : Location of upper threshold
        mu (float, Dist) : Mean of normal distribution
        sigma (float, Dist) : Standard deviation of normal distribution
    """
    dist = cores.truncnorm(lo, up, mu, sigma)
    dist.addattr(str="Truncnorm(%s,%s,%s,%s)"%(lo,up,mu,sigma))
    return dist


def Tukeylambda(shape=0, scale=1, shift=0):
    """
    Tukey-lambda distribution

    Args:
        lam (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.tukeylambda(shape)*scale + shift
    dist.addattr(str="Tukeylambda(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Uniform(lo=0, up=1):
    r"""
    Uniform distribution

    Args:
        lo (float, Dist) : Lower threshold of distribution. Must be smaller than up.
        up (float, Dist) : Upper threshold of distribution.

    Examples:
        >>> f = chaospy.Uniform(2, 4)
        >>> q = numpy.linspace(0,1,5)
        >>> print(numpy.around(f.inv(q), 4))
        [2.  2.5 3.  3.5 4. ]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.   0.25 0.5  0.75 1.  ]
        >>> print(numpy.around(f.sample(4), 4))
        [3.3072 2.23   3.9006 2.9644]
        >>> print(f.mom(1))
        3.0
    """

    dist = cores.uniform()*((up-lo)*.5)+((up+lo)*.5)
    dist.addattr(str="Uniform(%s,%s)"%(lo,up))
    return dist


def Wald(mu=0, scale=1, shift=0):
    """
    Wald distribution
    Reciprocal inverse Gaussian distribution

    Args:
        mu (float, Dist) : Mean of the normal distribution
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.wald(mu)*scale + shift
    dist.addattr(str="Wald(%s,%s,%s)"%(mu, scale, shift))
    return dist


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
    dist = cores.weibull(shape)*scale + shift
    dist.addattr(str="Weibull(%s,%s,%s)" % (shape, scale, shift))
    return dist


def Wigner(radius=1, shift=0):
    """
    Wigner (semi-circle) distribution

    Args:
        radius (float, Dist) : radius of the semi-circle (scale)
        shift (float, Dist) : location of the origen (location)
    """
    dist = radius*(2*cores.beta(1.5,1.5)-1) + shift
    dist.addattr(str="Wigner(%s,%s)" % (radius, shift))
    return dist


def Wrapcauchy(shape=0.5, scale=1, shift=0):
    """
    Wraped Cauchy distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = cores.wrapcauchy(shape)*scale + shift
    dist.addattr(str="Wrapcauchy(%s,%s,%s)"%(shape, scale, shift))
    return dist


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
    if lo is None:
        lo = samples.min()
    if up is None:
        up = samples.max()

    try:
        #construct the kernel density estimator
        kernel = gaussian_kde(samples, bw_method="scott")
        dist = cores.kdedist(kernel, lo, up)
        dist.addattr(str="SampleDist(%s,%s)" % (lo, up))

    #raised by gaussian_kde if dataset is singular matrix
    except numpy.linalg.LinAlgError:
        dist = Uniform(lo=-numpy.inf, up=numpy.inf)

    return dist

if __name__=='__main__':
    import __init__ as cp
    import numpy as np
    import doctest
    doctest.testmod()

