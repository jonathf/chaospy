"""
The superclass for Dist and tools for constructing a custom
distribuiton.

Each distribution have the following methods

fwd         Cumulative distribution function (Rosenblatt transform)
mom         Raw statistical moments
pdf         Probability density function
inv         Point percentile function (Inverse Rosenblatt)
sample      Random number sampler
ttr         Three terms recurrence coefficient generator
range       Upper and lower bounds of the distribution

If direct subset of Dist, the following method must be provided:

    _cdf(self, x, **prm)    Cumulative distribution function
    _bnd(self, **prm)       Upper and lower bounds

The following can be provided:

    _pdf(self, x, **prm)    Probability density function
    _ppf(self, q, **prm)    CDF inverse
    _mom(self, k, **prm)    Statistical moment generator
    _ttr(self, k, **prm)    TTR coefficients generator
    _str(self, **prm)       Preaty print of distribution

Alternative use the construct generator.

Examples
--------
A general uniform distribution wit two parameters
>>> class Uniform(pc.Dist):
...     def __init__(self, lo=0, up=1):
...         pc.Dist.__init__(self, lo=lo, up=up)
...     def _cdf(self, x, lo, up):
...         return (x-lo)/(up-lo)
...     def _bnd(self, lo, up):
...         return lo, up
...     def _pdf(self, x, lo, up):
...         return 1./(up-lo)
...     def _ppf(self, q, lo, up):
...         return q*(up-lo) + lo
...     def _str(self):
...         return "u(%s%s)" % (lo, up)
...
>>> dist = Uniform(-3,3)
>>> print dist.fwd([-3,0,3])
[ 0.   0.5  1. ]

See also
--------
For advanced variables, see dist.graph.
"""
import numpy as np
import new

from approx import pdf_full, inv, mom, find_interior_point

from graph import Graph
from sampler import samplegen

string = str # will be overridden locally
#operators imported at end

__all__ = [
"Dist", "construct"
        ]

class Dist(object):
    """
A represatation of a random variable.
    """

    def __init__(self, **prm):
        """
Parameters
----------
_length : int
    Length of the distribution
_advanced : bool
    If True, activate advanced mode
**prm : array_like
    Other optional parameters. Will be assumed when calling any
    sub-functions.
        """

        for key,val in prm.items():
            if not isinstance(val, Dist):
                prm[key] = np.array(val)

        self.length = prm.pop("_length", 1)
        self.advance = prm.pop("_advance", False)
        self.prm = prm.copy()
        self.G = Graph(self)
        self.dependencies = self.G.run(self.length, "dep")[0]

    def range(self, x=None, retall=False, verbose=False):
        """
Generate the upper and lower bounds of a distribution.

Parameters
----------
x : array_like, optional
    The bounds might vary over the sample space. By providing x you
    can specify where in the space the bound should be taken.
    If omited, a (pseudo-)random sample is used.

Returns
-------
out : np.ndarray
    The lower (out[0]) and upper (out[1]) bound where
    out.shape=(2,)+x.shape
        """

        dim = len(self)
        if x is None:
            x = find_interior_point(self)
        else:
            x = np.array(x)
        shape = x.shape
        size = x.size/dim
        x = x.reshape(dim, size)

        out, G = self.G.run(x, "range")
        out = out.reshape((2,)+shape)

        if verbose>1:
            print G

        if retall:
            return out, G
        return out

    def fwd(self, x, retall=False, verbose=False):
        """
Forward Rosenblatt transformation.

Parameters
----------
x : array_like
    Location for the distribution function. x.shape must be
    compatible with distribution shape.


Returns
-------
out : ndarray
    Evaluated distribution function values, where
    out.shape==x.shape.
        """
        dim = len(self)
        x = np.asfarray(x)
        shape = x.shape
        size = x.size/dim
        x = x.reshape(dim, size)

        bnd, G = self.G.run(x, "range")
        x_ = np.where(x<bnd[0], bnd[0], x)
        x_ = np.where(x_>bnd[1], bnd[1], x_)
        out, G = self.G.run(x_, "fwd")
        out = np.where(x<bnd[0], 0, out)
        out = np.where(x>bnd[1], 1, out)

        if verbose>1:
            print G

        out = out.reshape(shape)
        if retall:
            return out, G
        return out

    def inv(self, q, maxiter=100, tol=1e-5, verbose=False, **kws):
        """
Inverse Rosenblatt transformation

Parameters
----------
q : array_like
    Probabilities to be inverse. If any values are outside [0,1],
    error will be raised. q.shape must be compatible with
    diistribution shape.

Given that the analytical transformation isn't available, the
following parameters are used in the estimation.

maxiter : int
    Maximum number of iterations
tol : float
    Tolerence level

Returns
-------
out : ndarray
    Inverted probability values where out.shape==q.shape.
        """
        q = np.array(q)
        assert np.all(q>=0) and np.all(q<=1), q

        dim = len(self)
        shape = q.shape
        size = q.size/dim
        q = q.reshape(dim, size)

        try:
            out, G = self.G.run(q, "inv", maxiter=maxiter, tol=tol,
                    verbose=verbose)
        except NotImplementedError:
            out,N,q_ = inv(self, q,
                    maxiter=maxiter, tol=tol, retall=True)
            if verbose:
                diff = np.max(np.abs(q-q_))
                print "approx %s.inv w/%d calls and eps=%g" \
                        % (self, N, diff)

        lo,up = self.G.run(out, "range")[0]
        out = np.where(out.T>up.T, up.T, out.T).T
        out = np.where(out.T<lo.T, lo.T, out.T).T

        return out.reshape(shape)

    def pdf(self, x, step=1e-7, verbose=0):
        """
Probability density function.

Parameters
----------
x : array_like
    Location for the density function. x.shape must be compatible
    with distribution shape.
step : float, array_like
    The step length given aproximation is used. If array provided,
    elements are used along each axis.

Returns
-------
out : ndarray
    Evaluated density function values. Shapes are related through
    the identity x.shape=dist.shape+out.shape
        """
        dim = len(self)
        x = np.array(x)
        shape = x.shape
        size = x.size/dim
        x = x.reshape(dim, size)
        out = np.zeros((dim, size))

        (lo,up),G = self.G.run(x, "range")
        valids = np.prod((x.T>=lo.T)*(x.T<=up.T), 1, dtype=bool)
        x[:, True-valids] = (.5*(up+lo))[:, True-valids]
        out = np.zeros((dim,size))

        try:
            tmp,G = self.G.run(x, "pdf",
                    eps=step)
            out[:,valids] = tmp[:,valids]
        except NotImplementedError:
            tmp,G = pdf_full(self, x, step, retall=True)
            out[:,valids] = tmp[:,valids]
            if verbose:
                print "approx %s.pdf"
        except IndexError:
            pass

        if verbose>1:
            print self.G

        out = out.reshape(shape)
        if dim>1:
            out = np.prod(out, 0)
        return out

    def sample(self, shape=(), rule="R", antithetic=None,
            verbose=False, **kws):
        """
Create pseudo-random generated samples.

Parameters
----------
shape : array_like
    The shape of the samples to generate.
rule : str
    Alternative sampling techniques

    Normal sampling schemes
    Key     Name                Nested
    ----    ----------------    ------
    "K"     Korobov             no
    "R"     (Pseudo-)Random     no
    "L"     Latin hypercube     no
    "S"     Sobol               yes
    "H"     Halton              yes
    "M"     Hammersley          yes

    Grided sampling schemes
    Key     Name                Nested
    ----    ----------------    ------
    "C"     Chebyshev nodes     maybe
    "G"     Gaussian quadrature no
    "E"     Gauss-Legende nodes no

antithetic : bool, array_like
    If provided, will be used to setup antithetic variables.
    If array, defines the axes to mirror.

Returns
-------
out : ndarray
    Random samples with shape (len(self),)+self.shape
        """

        size = np.prod(shape, dtype=int)
        dim = len(self)
        if dim>1:
            if isinstance(shape, (tuple,list,np.ndarray)):
                shape = (dim,) + tuple(shape)
            else:
                shape = (dim, shape)

        out = samplegen(size, self, rule, antithetic)
        try:
            out = out.reshape(shape)
        except:
            if len(self)==1:
                out = out.flatten()
            else:
                out = out.reshape(dim, out.size/dim)


        return out



    def mom(self, K, **kws):
        """Raw statistical moments.

Creates non-centralized raw moments from the random variable. If
analytical options can not be utilized, Monte Carlo integration
will be used.

Parameters
----------
K : array_like
    Index of the raw moments. k.shape must be compatible with
    distribution shape.
    Sampling scheme when performing Monte Carlo
rule : str
    rule for estimating the moment if the analytical method
    fails.

    Key Monte Carlo schemes
    --- ------------------------
    "H" Halton sequence
    "K" Korobov set
    "L" Latin hypercube sampling
    "M" Hammersley sequence
    "R" (Pseudo-)Random sampling
    "S" Sobol sequence

    Key Quadrature schemes
    --- ------------------------
    "C" Clenshaw-Curtis
    "Q" Gaussian quadrature
    "E" Gauss-Legendre

composit : int, array_like optional
    If provided, composit quadrature will be used.
    Ignored in the case if gaussian=True.

    If int provided, determines number of even domain splits
    If array of ints, determines number of even domain splits along
        each axis
    If array of arrays/floats, determines location of splits

antithetic : array_like, optional
    List of bool. Represents the axes to mirror using antithetic
    variable during MCI.

Returns
-------
out : ndarray
    Shapes are related through the identity
    k.shape==dist.shape+k.shape
        """
        K = np.array(K, dtype=int)
        shape = K.shape
        dim = len(self)
        if dim>1:
            shape = shape[1:]
        size = K.size/dim
        K = K.reshape(dim, size)

        try:
            out, G = self.G.run(K, "mom", **kws)
        except NotImplementedError:
            out = mom(self, K, **kws)
        return out.reshape(shape)



    def ttr(self, k, acc=10**3, verbose=1):
        """Three terms relation's coefficient generator

Parameters
----------
k : array_like, int
    The order of the coefficients
acc : int
    Accuracy of discretized Stieltjes if analytical methods are
    unavailable.

Returns
-------
out : Recurrence coefficients
    Where out[0] is the first (A) and out[1] is the second coefficient
    With out.shape==(2,)+k.shape
        """

        k = np.array(k, dtype=int)
        dim = len(self)
        shape = k.shape
        shape = (2,) + shape
        size = k.size/dim
        k = k.reshape(dim, size)

        out, G = self.G.run(k, "ttr")
        return out.reshape(shape)

    def _ttr(self, *args, **kws):
        "Default TTR generator, throws error"
        raise NotImplementedError


    def _dep(self, G):
        """Default dependency module backend.
See graph for advanced distributions."""
        sets = [G(dist) for dist in G.D]
        if len(self)==1:
            out = [set([self])]
        else:
            out = [set([]) for _ in xrange(len(self))]
        for s in sets:
            for i in xrange(len(self)):
                out[i].update(s[i])

        return out


    def __str__(self):
        "X.__str__() <==> str(X)"
        if hasattr(self, "_str"):
            return string(self._str(**self.prm))
        return "D"

    def __len__(self):
        "X.__len__() <==> len(X)"
        return self.length

    def __add__(self, X):
        "Y.__add__(X) <==> X+Y"
        return add(self, X)

    def __radd__(self, X):
        "Y.__radd__(X) <==> Y+X"
        return add(X, self)

    def __sub__(self, X):
        "Y.__sub__(X) <==> X-Y"
        return add(self, -X)

    def __rsub__(self, X):
        "Y.__rsub__(X) <==> Y-X"
        return add(X, -self)

    def __neg__(self):
        "X.__neg__() <==> -X"
        return neg(self)

    def __mul__(self, X):
        "Y.__mul__(X) <==> X*Y"
        return mul(self, X)

    def __rmul__(self, X):
        "Y.__rmul__(X) <==> Y*X"
        return mul(X, self)

    def __div__(self, X):
        "Y.__div__(X) <==> Y/X"
        return mul(self, X**-1)

    def __rdiv__(self, X):
        "Y.__rdiv__(X) <==> X/Y"
        return mul(X, self**-1)

    def __pow__(self, X):
        "Y.__pow__(X) <==> Y**X"
        return pow(self, X)

    def __rpow__(self, X):
        "Y.__pow__(X) <==> X**Y"
        return pow(X, self)

    def __le__(self, X):
        "Y.__le__(X) <==> Y<=X"
        return trunk(self, X)

    def __lt__(self, X):
        "Y.__lt__(X) <==> Y<X"
        return trunk(self, X)

    def __ge__(self, X):
        "Y.__ge__(X) <==> Y>=X"
        return trunk(X, self)

    def __gt__(self, X):
        "Y.__gt__(X) <==> Y>X"
        return trunk(X, self)


    def addattr(self, **kws):
        """
Add attribution to distribution

Parameters
----------
pdf : callable
    Probability density function
cdf : callable
    Cumulative distribution function
ppf : callable
    Point percentile function
mom : callable
    Raw statistical moments
ttr : callable
    Three term recursion coefficient generator
val : callable
    If auxiliary variable, try to return the values of it's
    underlying variables, else return self.
str : callable, str
    Pretty print of module
dep : vallable
    Dependency structure (if non-trivial)
        """
        for key,val in kws.items():
            if key=="str" and isinstance(val, string):
                val_ = val
                val = lambda *a,**k: val_
            setattr(self, "_"+key, new.instancemethod(val, self, None))


    def dependent(self, *args):
        """
Determine dependency structure in module

Parameters
----------
*args : optional
    If omited, internal dependency will be determined.
    If included, dependency between self and args will be used.

Returns
-------
out : bool
    True if distribution is dependent.
        """

        sets, G = self.G.run(None, "dep")

        if args:
            sets = [reduce(lambda x,y: x.union(y), sets)]
            for arg in args:
                sets.append(reduce(lambda x,y: \
                        x.union(y), arg.G.run(None, "dep")))

        as_seperated = sum(map(len, sets))
        as_joined = len(reduce(lambda x,y: x.union(y), sets))
        return as_seperated!=as_joined


def construct(cdf, bnd, parent=None, pdf=None, ppf=None, mom=None,
        ttr=None, val=None, doc=None, str=None, dep=None,
        defaults=None, advance=False, length=1):
    """
Random variable constructor

Returns
-------
dist : Dist
    Distribution

Parameters
----------
cdf : callable
    Cumulative distribution function. Optional if parent is used.
bnd : callable
    Boundary interval. Optional if parent is used.
parent : Dist
    Distribution used as basis for new distribution. Any other
    argument that is omitted will instead take is function from
    parent.
doc : str, optional
    Documentation for the distribution.
str : str, callable, optional
    Pretty print of the variable
pdf : callable, optional
    Probability density function
ppf : callable, optional
    Point percentile function
mom : callable, optional
    Raw moment generator
ttr : callable, optional
    Three terms recursion coefficient generator
val : callable, optional
    Value function for transferable distributions.
dep : callable, optional
    Dependency structure.
advance : bool
    If True, advance mode is used. See dist.graph for details.
length : int
    If constructing an multivariate random variable, this sets the
    assumed length. Defaults to 1.
init : callable, optional

See also
--------
For simple use see dist.backend module
For advance use see dist.graph module
    """

    if not (parent is None):
        if hasattr(parent, "_cdf"):
            cdf = cdf or parent._cdf
        if hasattr(parent, "_bnd"):
            bnd = bnd or parent._bnd
        if hasattr(parent, "_pdf"):
            pdf = pdf or parent._pdf
        if hasattr(parent, "_ppf"):
            ppf = ppf or parent._ppf
        if hasattr(parent, "_mom"):
            mom = mom or parent._mom
        if hasattr(parent, "_ttr"):
            ttr = ttr or parent._ttr
        if hasattr(parent, "_str"):
            str = str or parent._str
        if hasattr(parent, "_dep"):
            dep = dep or parent._dep
        val = val or parent._val
        doc = doc or parent.__doc__

    def crash_func(*a, **kw):
        raise NotImplementedError
    if advance:
        ppf = ppf or crash_func
        pdf = pdf or crash_func
        mom = mom or crash_func
        ttr = ttr or crash_func

    def custom(**kws):

        if not (defaults is None):
            keys = defaults.keys()
            assert all([key in keys for key in kws.keys()])
            prm = defaults.copy()
        else:
            prm = {}
        prm.update(kws)
        _length = prm.pop("_length", length)
        _advance = prm.pop("_advance", advance)

        dist = Dist(_advance=_advance, _length=_length, **prm)

        dist.addattr(cdf=cdf)
        dist.addattr(bnd=bnd)

        if not (pdf is None):
            dist.addattr(pdf=pdf)
        if not (ppf is None):
            dist.addattr(ppf=ppf)
        if not (mom is None):
            dist.addattr(mom=mom)
        if not (ttr is None):
            dist.addattr(ttr=ttr)
        if not (val is None):
            dist.addattr(val=val)
        if not (str is None):
            dist.addattr(str=str)
        if not (dep is None):
            dist.addattr(dep=dep)

        return dist

    if not (doc is None):
        doc = """
Custom random variable
        """
    setattr(custom, "__doc__", doc)

    return custom


if __name__=='__main__':
    import __init__ as pc
    import numpy as np
    import doctest
    doctest.testmod()

from operators import add, mul, neg, pow, trunk

