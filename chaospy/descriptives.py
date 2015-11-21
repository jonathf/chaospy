"""
Collection of statistical descriptive tools
To be used together with Poly and Dist

Functions
---------
Acf         Auto-correlation function
Cov         Covariance function
Corr        Correlation function
E           Expected value operator
Kurt        Kurtosis operator
Perc        Percentile function
Skew        Skewness operator
Var         Variance function
"""
__version__ = "1.0"

def E(poly, dist=None, **kws):
    """
Expected value, or 1st order statistics of a probability
distribution or polynomial on a given probability space.

Parameters
----------
poly : Poly, Dist
    Input to take expected value on.
dist : Dist
    Defines the space the expected value is taken on.
    It is ignored if `poly` is a distribution.
**kws : optional
    Extra keywords passed to dist.mom.

Returns
-------
expected : ndarray
    The expected value of the polynomial or distribution, where
    `expected.shape==poly.shape`.

See Also
--------
Corr        Correlation matrix
Cov         Covariance matrix
Kurt        Kurtosis operator
Skew        Skewness operator
Var         Variance operator

Examples
--------
For distributions:
>>> x = cp.variable()
>>> Z = cp.Uniform()
>>> print cp.E(Z)
0.5
>>> print cp.E(x**3, Z)
0.25
    """
    if not isinstance(poly, (di.Dist, po.Poly)):
        print type(poly)
        print "Approximating expected value..."
        out = qu.quad(poly, dist, veceval=True, **kws)
        print "done"
        return out

    if isinstance(poly, di.Dist):
        dist = poly
        poly = po.variable(len(poly))

    if not poly.keys:
        return np.zeros(poly.shape, dtype=int)

    if isinstance(poly, (list, tuple, np.ndarray)):
        return [E(_, dist, **kws) for _ in poly]

    if poly.dim<len(dist):
        poly = po.setdim(poly, len(dist))

    shape = poly.shape
    poly = po.flatten(poly)

    keys = poly.keys
    mom = dist.mom(np.array(keys).T, **kws)
    A = poly.A

    if len(dist)==1:
        mom = mom[0]

    out = np.zeros(poly.shape)
    for i in xrange(len(keys)):
        out += A[keys[i]]*mom[i]

    out = np.reshape(out, shape)
    return out



def Var(poly, dist=None, **kws):
    """
Variance, or element by element 2nd order statistics of a
distribution or polynomial.

Parameters
----------
poly : Poly, Dist
    Input to take variance on.
dist : Dist
    Defines the space the variance is taken on.
    It is ignored if `poly` is a distribution.
**kws : optional
    Extra keywords passed to dist.mom.

Returns
-------
variation : ndarray
    Element for element variance along `poly`, where
    `variation.shape==poly.shape`.

See Also
--------
Corr        Correlation matrix
Cov         Covariance matrix
E           Expected value
Kurt        Kurtosis operator
Skew        Skewness operator

Examples
--------
>>> x = cp.variable()
>>> Z = cp.Uniform()
>>> print cp.Var(Z)
0.0833333333333
>>> print cp.Var(x**3, Z)
0.0803571428571
    """

    if isinstance(poly, di.Dist):
        x = po.variable(len(poly))
        poly, dist = x, poly
    else:
        poly = po.Poly(poly)

    dim = len(dist)
    if poly.dim<dim:
        po.setdim(poly, dim)

    shape = poly.shape
    poly = po.flatten(poly)

    keys = poly.keys
    N = len(keys)
    A = poly.A

    keys1 = np.array(keys).T
    if dim==1:
        keys1 = keys1[0]
        keys2 = sum(np.meshgrid(keys, keys))
    else:
        keys2 = np.empty((dim, N, N))
        for i in xrange(N):
            for j in xrange(N):
                keys2[:,i,j] = keys1[:,i]+keys1[:,j]

    m1 = np.outer(*[dist.mom(keys1, **kws)]*2)
    m2 = dist.mom(keys2, **kws)
    mom = m2-m1

    out = np.zeros(poly.shape)
    for i in xrange(N):
        a = A[keys[i]]
        out += a*a*mom[i,i]
        for j in xrange(i+1, N):
            b = A[keys[j]]
            out += 2*a*b*mom[i,j]

    out = out.reshape(shape)
    return out

def Std(poly, dist=None, **kws):
    """
Standard deviation, or element by element 2nd order statistics of a
distribution or polynomial.

Parameters
----------
poly : Poly, Dist
    Input to take variance on.
dist : Dist
    Defines the space the variance is taken on.
    It is ignored if `poly` is a distribution.
**kws : optional
    Extra keywords passed to dist.mom.

Returns
-------
variation : ndarray
    Element for element variance along `poly`, where
    `variation.shape==poly.shape`.

See Also
--------
Corr        Correlation matrix
Cov         Covariance matrix
E           Expected value
Kurt        Kurtosis operator
Skew        Skewness operator

Examples
--------
>>> x = cp.variable()
>>> Z = cp.Uniform()
>>> print cp.Var(Z)
0.0833333333333
>>> print cp.Var(x**3, Z)
0.0803571428571
    """

    if isinstance(poly, di.Dist):
        x = po.variable(len(poly))
        poly, dist = x, poly
    else:
        poly = po.Poly(poly)

    dim = len(dist)
    if poly.dim<dim:
        po.setdim(poly, dim)

    shape = poly.shape
    poly = po.flatten(poly)

    keys = poly.keys
    N = len(keys)
    A = poly.A

    keys1 = np.array(keys).T
    if dim==1:
        keys1 = keys1[0]
        keys2 = sum(np.meshgrid(keys, keys))
    else:
        keys2 = np.empty((dim, N, N))
        for i in xrange(N):
            for j in xrange(N):
                keys2[:,i,j] = keys1[:,i]+keys1[:,j]

    m1 = np.outer(*[dist.mom(keys1, **kws)]*2)
    m2 = dist.mom(keys2, **kws)
    mom = m2-m1

    out = np.zeros(poly.shape)
    for i in xrange(N):
        a = A[keys[i]]
        out += a*a*mom[i,i]
        for j in xrange(i+1, N):
            b = A[keys[j]]
            out += 2*a*b*mom[i,j]

    out = out.reshape(shape)
    return np.sqrt(out)



def Skew(poly, dist=None, **kws):
    """
Skewness, or element by element 3rd order statistics of a
distribution or polynomial.

Parameters
----------
poly : Poly, Dist
    Input to take skewness on.
dist : Dist
    Defines the space the skewness is taken on.
    It is ignored if `poly` is a distribution.
**kws : optional
    Extra keywords passed to dist.mom.

Returns
-------
skewness : ndarray
    Element for element variance along `poly`, where
    `skewness.shape==poly.shape`.

See Also
--------
Corr        Correlation matrix
Cov         Covariance matrix
E           Expected value
Kurt        Kurtosis operator
Var         Variance operator

Examples
--------
>>> x = cp.variable()
>>> Z = cp.Gamma()
>>> print cp.Skew(Z)
2.0
    """
    if isinstance(poly, di.Dist):
        x = po.variable(len(poly))
        poly, dist = x, poly
    else:
        poly = po.Poly(poly)

    if poly.dim<len(dist):
        po.setdim(poly, len(dist))

    shape = poly.shape
    poly = po.flatten(poly)

    m1 = E(poly, dist)
    m2 = E(poly**2, dist)
    m3 = E(poly**3, dist)
    out = (m3-3*m2*m1+2*m1**3)/(m2-m1**2)**1.5

    out = np.reshape(out, shape)
    return out



def Kurt(poly, dist=None, fisher=True, **kws):
    """
Kurtosis, or element by element 4rd order statistics of a
distribution or polynomial.

Parameters
----------
poly : Poly, Dist
    Input to take skewness on.
dist : Dist
    Defines the space the skewness is taken on.
    It is ignored if `poly` is a distribution.
fisher : bool
    If True, Fisher's definition is used (Normal -> 0.0). If False,
    Pearson's definition is used (normal -> 3.0)
**kws : optional
    Extra keywords passed to dist.mom.

Returns
-------
skewness : ndarray
    Element for element variance along `poly`, where
    `skewness.shape==poly.shape`.

See Also
--------
Corr        Correlation matrix
Cov         Covariance matrix
E           Expected value
Skew        Skewness operator
Var         Variance operator

Examples
--------
>>> x = cp.variable()
>>> Z = cp.Uniform()
>>> print cp.Kurt(Z)
-1.2
>>> Z = cp.Normal()
>>> print cp.Kurt(x, Z)
4.4408920985e-16
    """
    if isinstance(poly, di.Dist):
        x = po.variable(len(poly))
        poly, dist = x, poly
    else:
        poly = po.Poly(poly)

    if fisher: adjust = 3
    else: adjust = 0

    shape = poly.shape
    poly = po.flatten(poly)

    m1 = E(poly, dist)
    m2 = E(poly**2, dist)
    m3 = E(poly**3, dist)
    m4 = E(poly**4, dist)

    out = (m4-4*m3*m1 + 6*m2*m1**2 - 3*m1**4) /\
            (m2**2-2*m2*m1**2+m1**4) - adjust

    out = np.reshape(out, shape)
    return out

def Cov(poly, dist=None, **kws):
    """
Covariance matrix, or 2rd order statistics of a distribution or
polynomial.

Parameters
----------
poly : Poly, Dist
    Input to take covariance on. Must have `len(poly)>=2`.
dist : Dist
    Defines the space the covariance is taken on.
    It is ignored if `poly` is a distribution.
**kws : optional
    Extra keywords passed to dist.mom.

Returns
-------
covariance : ndarray
    Covariance matrix with
    `covariance.shape==poly.shape+poly.shape`.

See Also
--------
Corr        Correlation matrix
E           Expected value
Kurt        Kurtosis operator
Skew        Skewness operator
Var         Variance operator

Examples
--------
>>> Z = cp.MvNormal([0,0], [[2,.5],[.5,1]])
>>> print cp.Cov(Z)
[[ 2.   0.5]
 [ 0.5  1. ]]

>>> x = cp.variable()
>>> Z = cp.Normal()
>>> print cp.Cov([x, x**2], Z)
[[ 1.  0.]
 [ 0.  2.]]
    """
    if not isinstance(poly, (di.Dist, po.Poly)):
        poly = po.Poly(poly)

    if isinstance(poly, di.Dist):
        x = po.variable(len(poly))
        poly, dist = x, poly
    else:
        poly = po.Poly(poly)

    dim = len(dist)
    shape = poly.shape
    poly = po.flatten(poly)
    keys = poly.keys
    N = len(keys)
    A = poly.A
    keys1 = np.array(keys).T
    if dim==1:
        keys1 = keys1[0]
        keys2 = sum(np.meshgrid(keys, keys))
    else:
        keys2 = np.empty((dim, N, N))
        for i in xrange(N):
            for j in xrange(N):
                keys2[:, i,j] = keys1[:,i]+keys1[:,j]

    m1 = dist.mom(keys1, **kws)
    m2 = dist.mom(keys2, **kws)
    mom = m2-np.outer(m1, m1)

    out = np.zeros((len(poly), len(poly)))
    for i in xrange(len(keys)):
        a = A[keys[i]]
        out += np.outer(a,a)*mom[i,i]
        for j in xrange(i+1, len(keys)):
            b = A[keys[j]]
            ab = np.outer(a,b)
            out += (ab+ab.T)*mom[i,j]

    out = np.reshape(out, shape+shape)
    return out



def Corr(poly, dist=None, **kws):
    """
Correlation matrix of a distribution or polynomial.

Parameters
----------
poly : Poly, Dist
    Input to take correlation on. Must have `len(poly)>=2`.
dist : Dist
    Defines the space the correlation is taken on.
    It is ignored if `poly` is a distribution.
**kws : optional
    Extra keywords passed to dist.mom.

Returns
-------
correlation : ndarray
    Correlation matrix with
    `correlation.shape==poly.shape+poly.shape`.

See Also
--------
Acf         Auto-correlation function
Cov         Covariance matrix
E           Expected value
Kurt        Kurtosis operator
Skew        Skewness operator
Var         Variance operator

Examples
--------
>>> Z = cp.MvNormal([3,4], [[2,.5],[.5,1]])
>>> print cp.Corr(Z)
[[ 1.          0.35355339]
 [ 0.35355339  1.        ]]

>>> x = cp.variable()
>>> Z = cp.Normal()
>>> print cp.Corr([x, x**2], Z)
[[ 1.  0.]
 [ 0.  1.]]
    """
    if isinstance(poly, di.Dist):
        x = po.variable(len(poly))
        poly, dist = x, poly
    else:
        poly = po.Poly(poly)

    C = Cov(poly, dist, **kws)
    V = np.diag(C)
    S = np.sqrt(np.outer(V,V))
    return np.where(S>0, C/S, 0)


def Acf(poly, dist, N=None, **kws):
    """
Auto-correlation function

Parameters
----------
poly : Poly
    Polynomial of interest. Must have `len(poly)>N`
dist : Dist
    Defines the space the correlation is taken on.
N : int, optional
    The number of time steps appart included. If omited set to
    `len(poly)/2+1`
**kws : optional
    Extra keywords passed to dist.mom.

Returns
-------
Q : ndarray
    Auto-correlation of `poly` with `shape=(N,)`.
    Note that by definition `Q[0]=1`.

See Also
--------
Corr        Correlation matrix

Examples
--------
>>> poly = cp.prange(10)[1:]
>>> Z = cp.Uniform()
>>> print cp.Acf(poly, Z, 5)
[ 1.          0.99148391  0.9721971   0.94571181  0.91265479]
    """

    if N is None:
        N = len(poly)/2 + 1

    V = Corr(poly, dist, **kws)
    out = np.empty(N)

    for n in xrange(N):
        out[n] = np.mean(V.diagonal(n), 0)

    return out

def Spearman(poly, dist, sample=1e4, retall=False, **kws):
    """
Calculate Spearman's rank-order correlation coefficient

Parameters
----------
poly : Poly
    Polynomial of interest.
dist : Dist
    Defines the space where correlation is taken.
sample : int
    Number of samples used in estimation.
retall : bool
    If true, return p-value as well.
**kws : optional
    Extra keywords passed to dist.sample.

Returns
-------
rho[, p-value]

rho : float or ndarray
    Correlation output. Of type float if two-dimensional problem.
    Correleation matrix if larger.
p-value : float or ndarray
    The two-sided p-value for a hypothesis test whose null
    hypothesis is that two sets of data are uncorrelated, has same
    dimension as rho.
    """
    samples = dist.sample(sample, **kws)
    poly = po.flatten(poly)
    Y = poly(*samples)
    if retall:
        return spearmanr(Y.T)
    return spearmanr(Y.T)[0]


def Perc(poly, q, dist, sample=1e4, **kws):
    """
Percentile function

Parameters
----------
poly : Poly
    Polynomial of interest.
q : array_like
    positions where percentiles are taken. Must be a number or an
    array, where all values are on the interval `[0,100]`.
dist : Dist
    Defines the space where percentile is taken.
sample : int
    Number of samples used in estimation.
**kws : optional
    Extra keywords passed to dist.sample.

Returns
-------
Q : ndarray
    Percentiles of `poly` with `Q.shape=poly.shape+q.shape`.

Examples
--------
>>> cp.seed(1000)
>>> x,y = cp.variable(2)
>>> poly = cp.Poly([x, x*y])
>>> Z = cp.J(cp.Uniform(3,6), cp.Normal())
>>> print cp.Perc(poly, [0, 50, 100], Z)
[[  3.         -45.        ]
 [  4.5080777   -0.05862173]
 [  6.          45.        ]]
    """

    shape = poly.shape
    poly = po.flatten(poly)

    q = np.array(q)/100.
    dim = len(dist)

    # Interior
    sample = kws.pop("sample", 1e4)
    Z = dist.sample(sample, **kws)
    if dim==1:
        Z = (Z,)
        q = np.array([q])
    poly1 = poly(*Z)

    # Min/max
    mi, ma = dist.range().reshape(2,dim)
    ext = np.mgrid[(slice(0,2,1),)*dim].reshape(dim, 2**dim).T
    ext = np.where(ext, mi, ma).T
    poly2 = poly(*ext)
    poly2 = np.array([_ for _ in poly2.T if not np.any(np.isnan(_))]).T

    # Finish
    if poly2.shape:
        poly1 = np.concatenate([poly1,poly2], -1)
    samples = poly1.shape[-1]
    poly1.sort()
    out = poly1.T[np.asarray(q*(samples-1), dtype=int)]
    out = out.reshape(q.shape + shape)
    return out

def QoI_Dist(poly, dist, sample=1e4, **kws):
    """
TODO: write the documentation, find a good name 

Percentile function

Parameters
----------
poly : Poly
    Polynomial of interest.
q : array_like
    positions where percentiles are taken. Must be a number or an
    array, where all values are on the interval `[0,100]`.
dist : Dist
    Defines the space where percentile is taken.
sample : int
    Number of samples used in estimation.
**kws : optional
    Extra keywords passed to dist.sample.

Returns
-------
Q : ndarray
    Percentiles of `poly` with `Q.shape=poly.shape+q.shape`.

Examples
--------
>>> cp.seed(1000)
>>> x,y = cp.variable(2)
>>> poly = cp.Poly([x, x*y])
>>> Z = cp.J(cp.Uniform(3,6), cp.Normal())
>>> print cp.Perc(poly, [0, 50, 100], Z)
[[  3.         -45.        ]
 [  4.5080777   -0.05862173]
 [  6.          45.        ]]
    """
    #sample from the input dist
    samples = dist.sample(sample)
    
    qoi_dists = []
    numKdeCreationFailures = 0
    for i in range(0, len(poly)):
        #sample the polinomial solution
        dataset = poly[i](samples)
        
        try:
            #construct the kernel density estimator
            kernel = gaussian_kde(dataset, bw_method="scott")
            
            #construct the QoI distribution
            def eval_cdf(x, kernel):
                cdf_vals = np.zeros(x.shape)
                for i in range(0, len(x)):
                    cdf_vals[i] = [kernel.integrate_box_1d(0, x_i) for x_i in x[i]]
    
                return cdf_vals
                
            QoIDist = di.construct(
                cdf=lambda self,x,lo,up,kernel: eval_cdf(x, kernel[0]),
                bnd=lambda self,lo,up,kernel: (lo, up),
                pdf=lambda self,x,lo,up,kernel: kernel[0](x)
            )
    
            lo = np.min(dataset)
            up = np.max(dataset)
            qoi_dist = QoIDist(lo=lo, up=up, kernel=[kernel])
            
        except np.linalg.LinAlgError: #is raised by gaussian_kde if dataset is singular matrix
            qoi_dist = di.Uniform(lo=-np.inf, up=np.inf)
            numKdeCreationFailures = numKdeCreationFailures + 1
        
        qoi_dists.append(qoi_dist)
    
    if numKdeCreationFailures > 0:
        warn("num kde creation failures: " + str(numKdeCreationFailures))
    
    return qoi_dists

def E_cond(poly, freeze, dist, **kws):

    assert not dist.dependent()

    if poly.dim<len(dist):
        poly = po.setdim(poly, len(dist))

    freeze = po.Poly(freeze)
    freeze = po.setdim(freeze, len(dist))
    keys = freeze.A.keys()
    if len(keys)==1 and keys[0]==(0,)*len(dist):
        freeze = freeze.A.values()[0]
    else:
        freeze = np.array(keys)
    freeze = freeze.reshape(freeze.size/len(dist), len(dist))

    shape = poly.shape
    poly = po.flatten(poly)

    kmax = np.max(poly.keys, 0)+1
    keys = [i for i in np.ndindex(*kmax)]
    vals = dist.mom(np.array(keys).T, **kws).T
    mom = dict(zip(keys, vals))

    A = poly.A.copy()
    keys = A.keys()

    out = {}
    zeros = [0]*poly.dim
    for i in xrange(len(keys)):

        key = list(keys[i])
        a = A[tuple(key)]

        for d in xrange(poly.dim):
            for j in xrange(len(freeze)):
                if freeze[j,d]:
                    key[d], zeros[d] = zeros[d], key[d]
                    break

        tmp = a*mom[tuple(key)]
        if tuple(zeros) in out:
            out[tuple(zeros)] = out[tuple(zeros)] + tmp
        else:
            out[tuple(zeros)] = tmp

        for d in xrange(poly.dim):
            for j in xrange(len(freeze)):
                if freeze[j,d]:
                    key[d], zeros[d] = zeros[d], key[d]
                    break

    out = po.Poly(out, poly.dim, poly.shape, float)
    out = po.reshape(out, shape)

    return out

def Sens_m(poly, dist, **kws):
    """
Variance-based decomposition
AKA Sobol' indices

First order sensitivity indices
    """

    dim = len(dist)
    if poly.dim<dim:
        poly = po.setdim(poly, len(dist))

    zero = [0]*dim
    out = np.zeros((dim,) + poly.shape)
    V = Var(poly, dist, **kws)
    for i in range(dim):
        zero[i] = 1
        out[i] = Var(E_cond(poly, zero, dist, **kws), dist, **kws)/(V+(V==0))*(V!=0)
        zero[i] = 0
    return out


def Sens_m2(poly, dist, **kws):
    """
Variance-based decomposition
AKA Sobol' indices

Second order sensitivity indices
    """

    dim = len(dist)
    if poly.dim<dim:
        poly = po.setdim(poly, len(dist))

    zero = [0]*dim
    out = np.zeros((dim, dim) + poly.shape)
    V = Var(poly, dist, **kws)
    for i in range(dim):
        zero[i] = 1
        for j in range(dim):
            zero[j] = 1
            out[i] = Var(E_cond(poly, zero, dist, **kws), dist, **kws)/(V+(V==0))*(V!=0)
            zero[j] = 0
        zero[i] = 0

    return out


def Sens_t(poly, dist, **kws):
    """
Variance-based decomposition
AKA Sobol' indices

Total effect sensitivity index
    """

    dim = len(dist)
    if poly.dim<dim:
        poly = po.setdim(poly, len(dist))

    zero = [1]*dim
    out = np.zeros((dim,) + poly.shape, dtype=float)
    V = Var(poly, dist, **kws)
    for i in range(dim):
        zero[i] = 0
        out[i] = (V-Var(E_cond(poly, zero, dist, **kws),
            dist, **kws))/(V+(V==0))**(V!=0)
        zero[i] = 1
    return out

def Sens_m_nataf(order, dist, samples, vals, **kws):
    """
Variance-based decomposition with dependent varibles thorugh the Nataf
distribution.

First order sensitivity indices

Args:
    order (int): polynomial order used `orth_ttr`.
    dist (Copula): Assumed to be Nataf with independent components
    samples (array_like): Samples used for evaluation (typically generated from `dist`.)
    vals (array_like): Evaluations of the model for given samples.

Kwrds: Passed to `E`, `E_cond` and `Var` as part of the method.

Returns:
        np.ndarray: Sensitivity indices with `shape==(len(dist),)+vals.shape[1:]`
    """

    assert dist.__class__.__name__ == "Copula"
    trans = dist.prm["trans"]
    assert trans.__class__.__name__ == "nataf"
    vals = np.array(vals)

    cov = trans.prm["C"]
    cov = np.dot(cov, cov.T)

    marginal = dist.prm["dist"]
    dim = len(dist)

    orth = ort.orth_ttr(order, marginal, sort="GR")

    r = range(dim)

    index = [1] + [0]*(dim-1)

    nataf = di.Nataf(marginal, cov, r)
    samples_ = marginal.inv( nataf.fwd( samples ) )
    poly, coeffs = co.fit_regression(orth, samples_, vals, retall=1)

    V = Var(poly, marginal, **kws)

    out = np.zeros((dim,) + poly.shape)
    out[0] = Var(E_cond(poly, index, marginal, **kws), marginal, **kws)/(V+(V==0))*(V!=0)


    for i in xrange(1, dim):

        r = r[1:] + r[:1]
        index = index[-1:] + index[:-1]

        nataf = di.Nataf(marginal, cov, r)
        samples_ = marginal.inv( nataf.fwd( samples ) )
        poly, coeffs = co.fit_regression(orth, samples_, vals, retall=1)

        out[i] = Var(E_cond(poly, index, marginal, **kws), marginal, **kws)/(V+(V==0))*(V!=0)

    return out

def Sens_t_nataf(order, dist, samples, vals, **kws):
    """
Variance-based decomposition with dependent varibles thorugh the Nataf
distribution.

Total order sensitivity indices

Args:
    order (int): polynomial order used `orth_ttr`.
    dist (Copula): Assumed to be Nataf with independent components
    samples (array_like): Samples used for evaluation (typically generated from `dist`.)
    vals (array_like): Evaluations of the model for given samples.

Kwrds: Passed to `E`, `E_cond` and `Var` as part of the method.

Returns:
        np.ndarray: Sensitivity indices with `shape==(len(dist),)+vals.shape[1:]`
    """

    assert dist.__class__.__name__ == "Copula"
    trans = dist.prm["trans"]
    assert trans.__class__.__name__ == "nataf"
    vals = np.array(vals)

    cov = trans.prm["C"]
    cov = np.dot(cov, cov.T)

    marginal = dist.prm["dist"]
    dim = len(dist)

    orth = ort.orth_ttr(order, marginal, sort="GR")

    r = range(dim)

    index = [0] + [1]*(dim-1)

    nataf = di.Nataf(marginal, cov, r)
    samples_ = marginal.inv( nataf.fwd( samples ) )
    poly, coeffs = co.fit_regression(orth, samples_, vals, retall=1)

    V = Var(poly, marginal, **kws)

    out = np.zeros((dim,) + poly.shape)
    out[0] = (V-Var(E_cond(poly, index, marginal, **kws), marginal, **kws))/(V+(V==0))**(V!=0)


    for i in xrange(1, dim):

        r = r[1:] + r[:1]
        index = index[-1:] + index[:-1]

        nataf = di.Nataf(marginal, cov, r)
        samples_ = marginal.inv( nataf.fwd( samples ) )
        poly, coeffs = co.fit_regression(orth, samples_, vals, retall=1)

        out[i] = (V-Var(E_cond(poly, index, marginal, **kws), marginal, **kws))/(V+(V==0))*(V!=0)

    return out

def Sens_nataf(order, dist, samples, vals, **kws):
    """
Variance-based decomposition with dependent varibles thorugh the Nataf
distribution.

Main and total order sensitivity indices

Args:
    order (int): polynomial order used `orth_ttr`.
    dist (Copula): Assumed to be Nataf with independent components
    samples (array_like): Samples used for evaluation (typically generated from `dist`.)
    vals (array_like): Evaluations of the model for given samples.

Kwrds: Passed to `E`, `E_cond` and `Var` as part of the method.

Returns:
        np.ndarray: Sensitivity indices with `shape==(2,len(dist),)+vals.shape[1:]`.
        First component is main and second is total.
    """

    assert dist.__class__.__name__ == "Copula"
    trans = dist.prm["trans"]
    assert trans.__class__.__name__ == "nataf"
    vals = np.array(vals)

    cov = trans.prm["C"]
    cov = np.dot(cov, cov.T)

    marginal = dist.prm["dist"]
    dim = len(dist)

    orth = ort.orth_ttr(order, marginal, sort="GR")

    r = range(dim)

    index0 = [0] + [1]*(dim-1)
    index1 = [1] + [0]*(dim-1)

    nataf = di.Nataf(marginal, cov, r)
    samples_ = marginal.inv( nataf.fwd( samples ) )
    poly, coeffs = co.fit_regression(orth, samples_, vals, retall=1)

    V = Var(poly, marginal, **kws)

    out = np.zeros((2, dim,) + poly.shape)
    out[0,0] = (V-Var(E_cond(poly, index0, marginal, **kws), marginal, **kws))/(V+(V==0))**(V!=0)
    out[1,0] = Var(E_cond(poly, index1, marginal, **kws), marginal, **kws)/(V+(V==0))*(V!=0)


    for i in xrange(1, dim):

        r = r[1:] + r[:1]
        index0 = index0[-1:] + index0[:-1]

        nataf = di.Nataf(marginal, cov, r)
        samples_ = marginal.inv( nataf.fwd( samples ) )
        poly, coeffs = co.fit_regression(orth, samples_, vals, retall=1)

        out[0,i] = (V-Var(E_cond(poly, index0, marginal, **kws), marginal, **kws))/(V+(V==0))*(V!=0)
        out[1,i] = Var(E_cond(poly, index1, marginal, **kws), marginal, **kws)/(V+(V==0))*(V!=0)

    return out[::-1]

import numpy as np
import poly as po
import dist as di
import quadrature as qu
from scipy.stats import spearmanr, gaussian_kde
from warnings import warn

import collocation as co
import orthogonal as ort


if __name__=="__main__":
    import __init__ as cp
    import doctest
    doctest.testmod()
