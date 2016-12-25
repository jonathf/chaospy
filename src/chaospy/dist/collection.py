"""
Frontend for the collection distributions.

This modules provides a wrapper with documentation for the dist.cores module.
"""
import numpy as np
from scipy.stats import gaussian_kde

import chaospy.dist.cores
import chaospy.dist.joint


def Alpha(shape=1, scale=1, shift=0):
    """
    Alpha distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scale Parameter
        shift (float, Dist) : Location of lower threshold
    """
    dist = chaospy.dist.cores.alpha(shape)*scale + shift
    dist.addattr(str="Alpha(%s,%s,%s)" % (shape, scale, shift))
    return dist


def Anglit(loc=0, scale=1):
    """
    Anglit distribution.

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter
    """
    dist = chaospy.dist.cores.anglit()*scale + loc
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
    dist = chaospy.dist.cores.beta(shape, 1-shape)*(up-lo) + lo
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
        >>> cp.seed(1000)
        >>> f = cp.Beta(2, 2, 2, 3)
        >>> q = np.linspace(0,1,6)[1:-1]
        >>> print(f.inv(q))
        [ 2.28714073  2.43293108  2.56706892  2.71285927]
        >>> print(f.fwd(f.inv(q)))
        [ 0.2  0.4  0.6  0.8]
        >>> print(f.sample(4))
        [ 2.60388804  2.21123197  2.86505298  2.48812537]
        >>> print(f.mom(1))
        2.5
    """
    dist = chaospy.dist.cores.beta(a, b)*(up-lo) + lo
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
    dist = chaospy.dist.cores.bradford(c=shape)*(up-lo) + lo
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
    dist = chaospy.dist.cores.burr(c=1., d=1.)*scale + loc
    dist.addattr(str="Burr(%s,%s,%s,%s)"%(c, d, loc, scale))
    return dist


def Cauchy(loc=0, scale=1):
    """
    Cauchy distribution.

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter
    """
    dist = chaospy.dist.cores.cauchy()*scale + loc
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
    dist = chaospy.dist.cores.chi(df)*scale + shift
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
    dist = chaospy.dist.cores.chisquared(df, nc)*scale + shift
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
    dist = chaospy.dist.cores.dbl_gamma(shape)*scale + shift
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
    dist = chaospy.dist.cores.dbl_weibull(shape)*scale + shift
    dist.addattr(str="Dbl_weibull(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Exponential(scale=1, shift=0):
    R"""
    Exponential Probability Distribution

    Args:
        scale (float, Dist) : Scale parameter. scale!=0
        shift (float, Dist) : Location of the lower bound.

    Examples;:
        >>> cp.seed(1000)
        >>> f = cp.Exponential(1)
        >>> q = np.linspace(0,1,6)[1:-1]
        >>> print(f.inv(q))
        [ 0.22314355  0.51082562  0.91629073  1.60943791]
        >>> print(f.fwd(f.inv(q)))
        [ 0.2  0.4  0.6  0.8]
        >>> print(f.sample(4))
        [ 1.06013104  0.12217548  3.00140562  0.65814961]
        >>> print(f.mom(1))
        1.0
    """
    dist = chaospy.dist.cores.expon()*scale + shift
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
    dist = chaospy.dist.cores.exponpow(shape)*scale + shift
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
    dist = chaospy.dist.cores.exponweibull(a, c)*scale + shift
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
    dist = chaospy.dist.cores.f(n, m, nc)*scale + shift
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
    dist = chaospy.dist.cores.fatiguelife(shape)*scale + shift
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
    dist = chaospy.dist.cores.fisk(c=shape)*scale + shift
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
    dist = chaospy.dist.cores.foldcauchy(shape)*scale + shift
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
    dist = chaospy.dist.cores.foldnorm(mu-loc)*sigma + loc
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
    dist = chaospy.dist.cores.frechet(shape)*scale + shift
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
        >>> cp.seed(1000)
        >>> f = cp.Gamma(1, 1)
        >>> q = np.linspace(0,1,6)[1:-1]
        >>> print(f.inv(q))
        [ 0.22314355  0.51082562  0.91629073  1.60943791]
        >>> print(f.fwd(f.inv(q)))
        [ 0.2  0.4  0.6  0.8]
        >>> print(f.sample(4))
        [ 1.06013104  0.12217548  3.00140562  0.65814961]
        >>> print(f.mom(1))
        1.0
    """
    dist = chaospy.dist.cores.gamma(shape)*scale + shift
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
    dist = chaospy.dist.cores.genexpon(a=1, b=1, c=1)*scale + shift
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
    dist = chaospy.dist.cores.genextreme(shape)*scale + loc
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
    dist = chaospy.dist.cores.gengamma(shape1, shape2)*scale + shift
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
    dist = chaospy.dist.cores.genhalflogistic(shape)*scale + shift
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
    dist = chaospy.dist.cores.lognormal(1)*scale + shift
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
    dist = chaospy.dist.cores.gompertz(shape)*scale + shift
    dist.addattr(str="Gompertz(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Logweibul(scale=1, loc=0):
    """
    Gumbel or Log-Weibull distribution.

    Args:
        scale (float, Dist) : Scaling parameter
        loc (float, Dist) : Location parameter
    """
    dist = chaospy.dist.cores.gumbel()*scale + loc
    dist.addattr(str="Gumbel(%s,%s)"%(scale, loc))
    return dist


def Hypgeosec(loc=0, scale=1):
    """
    hyperbolic secant distribution

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scale parameter
    """
    dist = chaospy.dist.cores.hypgeosec()*scale + loc
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
    dist = chaospy.dist.cores.kumaraswamy(a,b)*(up-lo) + lo
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
        >>> cp.seed(1000)
        >>> f = cp.Laplace(2, 2)
        >>> q = np.linspace(0,1,6)[1:-1]
        >>> print(f.inv(q))
        [ 0.16741854  1.5537129   2.4462871   3.83258146]
        >>> print(f.fwd(f.inv(q)))
        [ 0.2  0.4  0.6  0.8]
        >>> print(f.sample(4))
        [ 2.73396771 -0.93923119  6.61651689  1.92746607]
        >>> print(f.mom(1))
        2.0
    """
    dist = chaospy.dist.cores.laplace()*scale + mu
    dist.addattr(str="Laplace(%s,%s)"%(mu,scale))
    return dist


def Levy(loc=0, scale=1):
    """
    Levy distribution

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter
    """
    dist = chaospy.dist.cores.levy()*scale+loc
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
    dist = chaospy.dist.cores.loggamma(shape)*scale + shift
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
    dist = chaospy.dist.cores.logistic()*scale + loc
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
    dist = chaospy.dist.cores.loglaplace(shape)*scale + shift
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
        >>> cp.seed(1000)
        >>> f = cp.Lognormal(0, 1)
        >>> q = np.linspace(0,1,6)[1:-1]
        >>> print(f.inv(q))
        [ 0.43101119  0.77619841  1.28833038  2.32012539]
        >>> print(f.fwd(f.inv(q)))
        [ 0.2  0.4  0.6  0.8]
        >>> print(f.sample(4))
        [ 1.48442856  0.30109692  5.19451094  0.95632796]
        >>> print(f.mom(1))
        1.6487212707
    """
    dist = chaospy.dist.cores.lognormal(sigma)*scale*np.e**mu + shift
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
    dist = chaospy.dist.cores.loguniform(lo, up)*scale + shift
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
    dist = chaospy.dist.cores.chi(3)*scale + shift
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
    dist = chaospy.dist.cores.mielke(kappa, expo)*scale + shift
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
    dist = chaospy.dist.cores.mvlognormal(loc, scale)
    dist.addattr(str="MvLognormal(%s,%s)" % (loc, scale))
    return dist


def MvNormal(loc=[0,0], scale=[[1,.5],[.5,1]]):
    """
    Multivariate Normal Distribution

    Args:
        loc (float, Dist) : Mean vector
        scale (float, Dist) : Covariance matrix or variance vector if scale is a 1-d vector.

    Examples:
        >>> cp.seed(1000)
        >>> f = cp.MvNormal([0,0], [[1,.5],[.5,1]])
        >>> q = [[.4,.5,.6],[.4,.5,.6]]
        >>> print(f.inv(q))
        [[-0.2533471   0.          0.2533471 ]
        [-0.34607858  0.          0.34607858]]
        >>> print(f.fwd(f.inv(q)))
        [[ 0.4  0.5  0.6]
        [ 0.4  0.5  0.6]]
        >>> print(f.sample(3))
        [[ 0.39502989 -1.20032309  1.64760248]
        [ 0.15884312  0.38551963  0.1324068 ]]
        >>> print(f.mom((1,1)))
        0.5
    """
    if np.all((np.diag(np.diag(scale))-scale)==0):
        out = chaospy.dist.joint.J(
            *[Normal(loc[i], scale[i,i]) for i in range(len(scale))])
    else:
        out = chaospy.dist.cores.mvnormal(loc, scale)
    out.addattr(str="MvNormal(%s,%s)" % (loc, scale))
    return out


def MvStudent_t(df=1, loc=[0,0], scale=[[1,.5],[.5,1]]):
    """
    Args:
        df (float, Dist) : Degree of freedom
        loc (array_like, Dist) : Location parameter
        scale (array_like) : Covariance matrix
    """
    out = chaospy.dist.cores.mvstudentt(df, loc, scale)
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
    dist = chaospy.dist.cores.nakagami(shape)*scale + shift
    dist.addattr(str="Nakagami(%s,%s,%s)"%(shape,scale,shift))
    return dist


def Normal(mu=0, sigma=1):
    R"""
    Normal (Gaussian) distribution

    Args:
        mu (float, Dist) : Mean of the distribution.
        sigma (float, Dist) : Standard deviation.  sigma > 0

    Examples:
        >>> cp.seed(1000)
        >>> f = cp.Normal(2, 2)
        >>> q = np.linspace(0,1,6)[1:-1]
        >>> print(f.inv(q))
        [ 0.31675753  1.49330579  2.50669421  3.68324247]
        >>> print(f.fwd(f.inv(q)))
        [ 0.2  0.4  0.6  0.8]
        >>> print(f.sample(4))
        [ 2.79005978 -0.40064618  5.29520496  1.91069125]
        >>> print(f.mom(1))
        2.0
    """
    dist = chaospy.dist.cores.normal()*sigma + mu
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
    dist = chaospy.dist.cores.pareto1(shape)*scale + loc
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
    dist = chaospy.dist.cores.pareto(shape)*scale + loc
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
    dist = chaospy.dist.cores.beta(shape, 1)*(up-lo) + lo
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
    dist = chaospy.dist.cores.powerlognorm(shape, sigma)*scale*np.e**mu + shift
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
    dist = chaospy.dist.cores.powernorm(shape)*scale + mu
    dist.addattr(str="Powernorm(%s,%s,%s)"%(shape, mu, scale))
    return dist


def Raised_cosine(loc=0, scale=1):
    """
    Raised cosine distribution

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scale parameter
    """
    dist = chaospy.dist.cores.raised_cosine()*scale + loc
    dist.addattr(str="Raised_cosine(%s,%s)"%(loc,scale))
    return dist


def Rayleigh(scale=1, shift=0):
    """
    Rayleigh distribution

    Args:
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = chaospy.dist.cores.chi(2)*scale + shift
    dist.addattr(str="Rayleigh(%s,%s)"%(scale, shift))
    return dist


def Reciprocal(lo=1, up=2):
    """
    Reciprocal distribution

    Args:
        lo (float, Dist) : Location of lower threshold
        up (float, Dist) : Location of upper threshold
    """
    dist = chaospy.dist.cores.reciprocal(lo,up)
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
    dist = chaospy.dist.cores.student_t(df)*scale + loc
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
        >>> cp.seed(1000)
        >>> f = cp.Triangle(2, 3, 4)
        >>> q = np.linspace(0,1,6)[1:-1]
        >>> print(f.inv(q))
        [ 2.63245553  2.89442719  3.10557281  3.36754447]
        >>> print(f.fwd(f.inv(q)))
        [ 0.2  0.4  0.6  0.8]
        >>> print(f.sample(4))
        [ 3.16764141  2.47959763  3.684668    2.98202994]
        >>> print(f.mom(1))
        3.0
    """
    dist = chaospy.dist.cores.triangle((mid-lo)*1./(up-lo))*(up-lo) + lo
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
    dist = chaospy.dist.cores.truncexpon((up-shift)/scale)*scale + shift
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
    dist = chaospy.dist.cores.truncnorm(lo, up, mu, sigma)
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
    dist = chaospy.dist.cores.tukeylambda(shape)*scale + shift
    dist.addattr(str="Tukeylambda(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Uniform(lo=0, up=1):
    r"""
    Uniform distribution

    Args:
        lo (float, Dist) : Lower threshold of distribution. Must be smaller than up.
        up (float, Dist) : Upper threshold of distribution.

    Examples:
        >>> cp.seed(1000)
        >>> f = cp.Uniform(2, 4)
        >>> q = np.linspace(0,1,5)
        >>> print(f.inv(q))
        [ 2.   2.5  3.   3.5  4. ]
        >>> print(f.fwd(f.inv(q)))
        [ 0.    0.25  0.5   0.75  1.  ]
        >>> print(f.sample(4))
        [ 3.30717917  2.23001389  3.90056573  2.9643828 ]
        >>> print(f.mom(1))
        3.0
    """

    dist = chaospy.dist.cores.uniform()*((up-lo)*.5)+((up+lo)*.5)
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
    dist = chaospy.dist.cores.wald(mu)*scale + shift
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
        >>> cp.seed(1000)
        >>> f = cp.Weibull(2)
        >>> q = np.linspace(0,1,6)[1:-1]
        >>> print(f.inv(q))
        [ 0.47238073  0.71472066  0.95723076  1.26863624]
        >>> print(f.fwd(f.inv(q)))
        [ 0.2  0.4  0.6  0.8]
        >>> print(f.sample(4))
        [ 1.02962665  0.34953609  1.73245653  0.8112642 ]
        >>> print(f.mom(1))
        0.886226925453
    """
    dist = chaospy.dist.cores.weibull(shape)*scale + shift
    dist.addattr(str="Weibull(%s,%s,%s)" % (shape, scale, shift))
    return dist


def Wigner(radius=1, shift=0):
    """
    Wigner (semi-circle) distribution

    Args:
        radius (float, Dist) : radius of the semi-circle (scale)
        shift (float, Dist) : location of the origen (location)
    """
    dist = radius*(2*chaospy.dist.cores.beta(1.5,1.5)-1) + shift
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
    dist = chaospy.dist.cores.wrapcauchy(shape)*scale + shift
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
        dist = chaospy.dist.cores.kdedist(kernel, lo, up)
        dist.addattr(str="SampleDist(%s,%s)" % (lo, up))

    #raised by gaussian_kde if dataset is singular matrix
    except np.linalg.LinAlgError:
        dist = Uniform(lo=-np.inf, up=np.inf)

    return dist

if __name__=='__main__':
    import __init__ as cp
    import numpy as np
    import doctest
    doctest.testmod()

