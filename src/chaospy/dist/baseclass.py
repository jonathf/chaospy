"""
Constructing custom probability distributions is done in one of two ways:
Sublcassing the :class:`~chaospy.dist.Dist` or by calling
:func:`~chaospy.dist.construct`. They work about the same except for one
methods are defined, while in the other, functions.

A simple example for constructing a simple uniform distribution::

    >>> def cdf(self, x, lo, up):
    ...     return (x-lo)/(up-lo)
    >>> def bnd(self, lo, up):
    ...     return lo, up
    >>> Uniform = cp.construct(cdf=cdf, bnd=bnd)
    >>> dist = Uniform(-3, 3)
    >>> print(dist.fwd([-3, 0, 3]))
    [ 0.   0.5  1. ]

Here ``cdf`` is the dependent cumulative distribution function as defined in
equation , ``bnd`` is a function returning the lower and upper bounds, and
``a`` and ``b`` are distribution parameters.  They take either other
components, or as illustrated: constants.


In addition to ``cdf`` and ``bnd`` there are a few optional arguments. For
example a fully featured uniform random variable is defined as follows::

    >>> def pdf(self, x, lo, up):
    ...     return 1./(up-lo)
    >>> def ppf(self, q, lo, up):
    ...     return q*(up-lo) + lo
    >>> Uniform = cp.construct(
    ...     cdf=cdf, bnd=bnd, pdf=pdf, ppf=ppf)
    >>> dist = Uniform(-3, 3)

There ``pdf`` is probability distribution function and ``ppf`` if the point
percentile function.  These are methods that provides needed functionality for
probabilistic collocation. If they are not provided during construct, they are
estimated as far as possible.

Equivalently constructing the same distribution using subclass:
:func:`~chaospy.dist.construct`::

    >>> class Uniform(cp.Dist):
    ...     def __init__(self, lo=0, up=1):
    ...         cp.Dist.__init__(self, lo=lo, up=up)
    ...     def _cdf(self, x, lo, up):
    ...         return (x-lo)/(up-lo)
    ...     def _bnd(self, lo, up):
    ...         return lo, up
    ...     def _pdf(self, x, lo, up):
    ...         return 1./(up-lo)
    ...     def _ppf(self, q, lo, up):
    ...         return q*(up-lo) + lo
    ...     def _str(self, lo, up):
    ...         return "u(%s%s)" % (lo, up)
    >>> dist = Uniform(-3, 3)
    >>> print(dist.fwd([-3, 0, 3])) # Forward Rosenblatt transformation
    [ 0.   0.5  1. ]

"""
import types
import numpy as np

import chaospy.dist


class Dist(object):
    """
    The distribution backend class.

    Subclass this module to construct a custom distribuiton.

    If direct subclass of Dist, two method must be provided:

    * Cumulative distribution function (CDF): ``_cdf(self, x, **prm)``.
    * Upper and lower bounds ``_bnd(self, **prm)``.

    The following can be provided:

    * Probability density function: ``_pdf(self, x, **prm)``.
    * CDF inverse: ``_ppf(self, q, **prm)``.
    * Statistical moment generator: ``_mom(self, k, **prm)``.
    * TTR coefficients generator: ``_ttr(self, k, **prm)``.
    * Pretty print of distribution: ``_str(self, **prm)``.

    Alternative use the construct generator :func:`~chaospy.dist.construct`.
    """

    __array_priority__ = 9000

    def __init__(self, **prm):
        """
        Args:
            _length (int) : Length of the distribution
            _advanced (bool) : If True, activate advanced mode
            **prm (array_like) : Other optional parameters. Will be assumed when
                    calling any sub-functions.
        """
        for key, val in prm.items():
            if not isinstance(val, Dist):
                prm[key] = np.array(val)

        self.length = int(prm.pop("_length", 1))
        self.advance = prm.pop("_advance", False)
        self.prm = prm.copy()
        self.graph = chaospy.dist.graph.Graph(self)
        self.dependencies = self.graph.run(self.length, "dep")[0]

    def range(self, x=None, retall=False, verbose=False):
        """
        Generate the upper and lower bounds of a distribution.

        Args:
            x (array_like, optional) : The bounds might vary over the sample
                    space. By providing x you can specify where in the space
                    the bound should be taken.  If omited, a (pseudo-)random
                    sample is used.

        Returns:
            (np.ndarray) : The lower (out[0]) and upper (out[1]) bound where
                    out.shape=(2,)+x.shape
        """
        dim = len(self)
        if x is None:
            x = chaospy.dist.approx.find_interior_point(self)
        else:
            x = np.array(x)
        shape = x.shape
        size = int(x.size/dim)
        x = x.reshape(dim, size)

        out, graph = self.graph.run(x, "range")
        out = out.reshape((2,)+shape)

        if verbose>1:
            print(graph)

        if retall:
            return out, graph
        return out

    def fwd(self, x):
        """
        Forward Rosenblatt transformation.

        Args:
            x (array_like) : Location for the distribution function. x.shape
                    must be compatible with distribution shape.

        Returns:
            (ndarray) : Evaluated distribution function values, where
                    out.shape==x.shape.
        """
        return chaospy.dist.rosenblatt.fwd(self, x)

    def cdf(self, x):
        """
        Cumulative distribution function.

        Note that chaospy only supports cumulative distribution funcitons in
        one dimensions.

        Args:
            x (array_like) : Location for the distribution function. x.shape
                    must be compatible with distribution shape.

        Returns:
            (ndarray) : Evaluated distribution function values, where
                    out.shape==x.shape.

        Except:
            (NotImplementedError) : for distributions with dependent
                components.
        """
        if self.dependent():
            raise NotImplementedError("""\
Cumulative distribution function is only available for stocastically \
independent variables""")
        out = chaospy.dist.rosenblatt.fwd(self, x)
        if len(self) > 1:
            out = np.prod(out, 0)
        return out


    def inv(self, q, maxiter=100, tol=1e-5, verbose=False, **kws):
        """
        Inverse Rosenblatt transformation.

        Args:
            q (array_like) : Probabilities to be inverse. If any values are
                    outside [0,1], error will be raised. q.shape must be
                    compatible with diistribution shape.

        Kwargs:
            maxiter (int) : Maximum number of iterations
            tol (float) : Tolerence level

        Returns:
            (ndarray) : Inverted probability values where out.shape==q.shape.
        """
        return chaospy.dist.rosenblatt.inv(self, q, maxiter, tol, **kws)

    def pdf(self, x, step=1e-7, verbose=0):
        """
        Probability density function.

        Args:
            x (array_like) : Location for the density function. x.shape must
                    be compatible with distribution shape.
            step (float, array_like) : The step length given aproximation is
                    used. If array provided, elements are used along each
                    axis.

        Returns:
            (ndarray) : Evaluated density function values. Shapes are related
                    through the identity x.shape=dist.shape+out.shape
        """
        dim = len(self)
        x = np.array(x)
        shape = x.shape
        size = int(x.size/dim)
        x = x.reshape(dim, size)
        out = np.zeros((dim, size))

        (lo, up), graph = self.graph.run(x, "range")
        valids = np.prod((x.T >= lo.T)*(x.T <= up.T), 1, dtype=bool)
        x[:, ~valids] = (.5*(up+lo))[:, ~valids]
        out = np.zeros((dim,size))

        try:
            tmp,graph = self.graph.run(x, "pdf",
                    eps=step)
            out[:,valids] = tmp[:,valids]
        except NotImplementedError:
            tmp,graph = chaospy.dist.approx.pdf_full(self, x, step, retall=True)
            out[:,valids] = tmp[:,valids]
            if verbose:
                print("approx %s.pdf")
        except IndexError:
            pass

        if verbose>1:
            print(self.graph)

        out = out.reshape(shape)
        if dim>1:
            out = np.prod(out, 0)
        return out

    def sample(self, size=(), rule="R", antithetic=None,
            verbose=False, **kws):
        """
        Create pseudo-random generated samples.

        Args:
            size (int,array_like) : The size of the samples to generate.
            rule (str) : Alternative sampling techniques. See
                    :func:`~chaospy.dist.samplegen`.
            antithetic (bool, array_like) : If provided, will be used to setup
                    antithetic variables.  If array, defines the axes to mirror.

        Returns:
            (ndarray) : Random samples with shape (len(self),)+self.shape
        """
        size_ = np.prod(size, dtype=int)
        dim = len(self)
        if dim>1:
            if isinstance(size, (tuple,list,np.ndarray)):
                shape = (dim,) + tuple(size)
            else:
                shape = (dim, size)
        else:
            shape = size

        out = chaospy.dist.sampler.samplegen(size_, self, rule, antithetic)
        try:
            out = out.reshape(shape)
        except:
            if len(self)==1:
                out = out.flatten()
            else:
                out = out.reshape(dim, int(out.size/dim))

        return out

    def mom(self, K, **kws):
        """
        Raw statistical moments.

        Creates non-centralized raw moments from the random variable. If
        analytical options can not be utilized, Monte Carlo integration
        will be used.

        Args:
            K (array_like) : Index of the raw moments. k.shape must be
                    compatible with distribution shape.  Sampling scheme when
                    performing Monte Carlo
            rule (str) : rule for estimating the moment if the analytical
                    method fails.
            composit (int, array_like optional) : If provided, composit
                    quadrature will be used.  Ignored in the case if
                    gaussian=True.  If int provided, determines number of even
                    domain splits. If array of ints, determines number of even
                    domain splits along each axis. If array of arrays/floats,
                    determines location of splits.
            antithetic (array_like, optional) : List of bool. Represents the
                    axes to mirror using antithetic variable during MCI.

        Returns:
            (ndarray) : Shapes are related through the identity
                    `k.shape==dist.shape+k.shape`.
        """
        K = np.array(K, dtype=int)
        shape = K.shape
        dim = len(self)

        if dim > 1:
            shape = shape[1:]

        size = int(K.size/dim)
        K = K.reshape(dim, size)

        try:
            out, graph = self.graph.run(K, "mom", **kws)

        except NotImplementedError:
            out = chaospy.dist.approx.mom(self, K, **kws)

        return out.reshape(shape)

    def ttr(self, k, acc=10**3, verbose=1):
        """
        Three terms relation's coefficient generator

        Args:
            k (array_like, int) : The order of the coefficients.
            acc (int) : Accuracy of discretized Stieltjes if analytical
                    methods are unavailable.

        Returns:
            (Recurrence coefficients) : Where out[0] is the first (A) and
                    out[1] is the second coefficient With
                    `out.shape==(2,)+k.shape`.
        """
        k = np.array(k, dtype=int)
        dim = len(self)
        shape = k.shape
        shape = (2,) + shape
        size = int(k.size/dim)
        k = k.reshape(dim, size)

        out, graph = self.graph.run(k, "ttr")
        return out.reshape(shape)

    def _ttr(self, *args, **kws):
        """Default TTR generator, throws error."""
        raise NotImplementedError

    def _dep(self, graph):
        """
        Default dependency module backend.

        See graph for advanced distributions.
        """
        sets = [graph(dist) for dist in graph.dists]
        if len(self)==1:
            out = [set([self])]
        else:
            out = [set([]) for _ in range(len(self))]
        for s in sets:
            for i in range(len(self)):
                out[i].update(s[i])

        return out


    def __str__(self):
        """X.__str__() <==> str(X)"""
        if hasattr(self, "_str"):
            return str(self._str(**self.prm))
        return "D"

    def __len__(self):
        """X.__len__() <==> len(X)"""
        return self.length

    def __add__(self, X):
        """Y.__add__(X) <==> X+Y"""
        return chaospy.dist.operators.add(self, X)

    def __radd__(self, X):
        """Y.__radd__(X) <==> Y+X"""
        return chaospy.dist.operators.add(self, X)

    def __sub__(self, X):
        """Y.__sub__(X) <==> X-Y"""
        return chaospy.dist.operators.add(self, -X)

    def __rsub__(self, X):
        """Y.__rsub__(X) <==> Y-X"""
        return chaospy.dist.operators.add(X, -self)

    def __neg__(self):
        """X.__neg__() <==> -X"""
        return chaospy.dist.operators.neg(self)

    def __mul__(self, X):
        """Y.__mul__(X) <==> X*Y"""
        return chaospy.dist.operators.mul(self, X)

    def __rmul__(self, X):
        """Y.__rmul__(X) <==> Y*X"""
        return chaospy.dist.operators.mul(self, X)

    def __div__(self, X):
        """Y.__div__(X) <==> Y/X"""
        return chaospy.dist.operators.mul(self, X**-1)

    def __rdiv__(self, X):
        """Y.__rdiv__(X) <==> X/Y"""
        return chaospy.dist.operators.mul(X, self**-1)

    def __floordiv__(self, X):
        """Y.__floordiv__(X) <==> Y/X"""
        return chaospy.dist.operators.mul(self, X**-1)

    def __rfloordiv__(self, X):
        """Y.__rfloordiv__(X) <==> X/Y"""
        return chaospy.dist.operators.mul(X, self**-1)

    def __truediv__(self, X):
        """Y.__truediv__(X) <==> Y/X"""
        return chaospy.dist.operators.mul(self, X**-1)

    def __rtruediv__(self, X):
        """Y.__rtruediv__(X) <==> X/Y"""
        return chaospy.dist.operators.mul(X, self**-1)

    def __pow__(self, X):
        """Y.__pow__(X) <==> Y**X"""
        return chaospy.dist.operators.pow(self, X)

    def __rpow__(self, X):
        """Y.__rpow__(X) <==> X**Y"""
        return chaospy.dist.operators.pow(X, self)

    def __le__(self, X):
        """Y.__le__(X) <==> Y<=X"""
        return chaospy.dist.operators.trunk(self, X)

    def __lt__(self, X):
        """Y.__lt__(X) <==> Y<X"""
        return chaospy.dist.operators.trunk(self, X)

    def __ge__(self, X):
        """Y.__ge__(X) <==> Y>=X"""
        return chaospy.dist.operators.trunk(X, self)

    def __gt__(self, X):
        """Y.__gt__(X) <==> Y>X"""
        return chaospy.dist.operators.trunk(X, self)

    def addattr(self, **kws):
        """
        Add attribution to distribution

        Kwargs:
            pdf (callable) : Probability density function.
            cdf (callable) : Cumulative distribution function.
            ppf (callable) : Point percentile function.
            mom (callable) : Raw statistical moments.
            ttr (callable) : Three term recursion coefficient generator.
            val (callable) : If auxiliary variable, try to return the values
                    of it's underlying variables, else return self.
            str (callable, str) : Pretty print of module.
            dep (callable) : Dependency structure (if non-trivial).
        """
        for key,val in kws.items():
            if key=="str" and isinstance(val, str):
                val_ = val
                val = lambda *a,**k: val_
            setattr(self, "_"+key, types.MethodType(val, self))


    def dependent(self, *args):
        """
        Determine dependency structure in module

        Args:
            arg, [...] (optional) : If omited, internal dependency will be
                    determined.  If included, dependency between self and args
                    will be used.

        Returns:
            (bool) : True if distribution is dependent.
        """
        sets, graph = self.graph.run(None, "dep")

        if args:

            sets_ = set()
            for set_ in sets:
                sets_ = sets_.union(set_)
            sets = [sets_]

            for arg in args:
                sets_ = set()
                for set_ in arg.graph.run(None, "dep"):
                    sets_ = sets_.union(set_)
                sets.append(sets_)

        as_seperated = sum([len(set_) for set_ in sets])

        as_joined = set()
        for set_ in sets:
            as_joined = as_joined.union(set_)
        as_joined = len(as_joined)

        return as_seperated != as_joined


def construct(cdf, bnd, parent=None, pdf=None, ppf=None, mom=None, ttr=None,
              val=None, doc=None, str=None, dep=None, defaults=None,
              advance=False, length=1):
    """
    Random variable constructor.

    Args:
        cdf (callable) : Cumulative distribution function. Optional if parent
                is used.
        bnd (callable) : Boundary interval. Optional if parent is used.
        parent (Dist) : Distribution used as basis for new distribution. Any
                other argument that is omitted will instead take is function
                from parent.
        doc (str, optional) : Documentation for the distribution.
        str (str, callable, optional) : Pretty print of the variable.
        pdf (callable, optional) : Probability density function.
        ppf (callable, optional) : Point percentile function.
        mom (callable, optional) : Raw moment generator.
        ttr (callable, optional) : Three terms recursion coefficient generator
        val (callable, optional) : Value function for transferable
                distributions.
        dep (callable, optional) : Dependency structure.
        advance (bool) : If True, advance mode is used. See dist.graph for
                details.
        length (int) : If constructing an multivariate random variable, this
                sets the assumed length. Defaults to 1.
        init (callable, optional) : Custom constructor method.

    Returns:
        dist (Dist) : New custom distribution.
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
