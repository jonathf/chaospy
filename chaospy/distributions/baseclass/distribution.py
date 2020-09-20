"""Abstract baseclass for all distributions."""
import logging
import abc
import six
from itertools import permutations
import numpy

import chaospy

DISTRIBUTION_IDENTIFIERS = {}


@six.add_metaclass(abc.ABCMeta)
class Distribution():
    """Baseclass for all probability distributions."""

    __array_priority__ = 9000
    """Numpy override variable."""

    interpret_as_integer = False
    """
    Flag indicating that return value from the methods sample, and inv
    should be interpreted as integers instead of floating point.
    """

    @property
    def stochastic_dependent(self):
        """True if distribution contains stochastically dependent components."""
        return any(len(deps) > 1 for deps in self._dependencies)

    def shares_dependencies(self, *others):
        distributions = [self]+[other for other in others
                                if isinstance(other, Distribution)]
        if len(distributions) == 1:
            return False
        dependencies = [{dep for deps in dist._dependencies for dep in deps}
                        for dist in distributions]
        for deps1, deps2 in permutations(dependencies, 2):
            if deps1.intersection(deps2):
                return True
        return False

    def __init__(
            self,
            parameters,
            dependencies,
            rotation=None,
            exclusion=None,
            repr_args=None,
            index_cls=None,
    ):
        """
        Distribution initializer.

        In addition to assigning some object variables, also checks for
        some consistency issues.

        Args:
            parameters (Optional[Distribution[str, Union[ndarray, Distribution]]]):
                Collection of model parameters.
            dependencies (Optional[Sequence[Set[int]]]):
                Dependency identifiers. One collection for each dimension.
            rotation (Optional[Sequence[int]]):
                The order of which to resolve dependencies.
            exclusion (Optional[Sequence[int]]):
                Distributions that has been "taken out of play" and
                therefore can not be reused other places in the
                dependency hierarchy.
            repr_args (Optional[Sequence[str]]):
                Positional arguments to place in the object string
                representation. The repr output will then be:
                `<class name>(<arg1>, <arg2>, ...)`.
            index_cls (Optional[Type[Index]]):
                Class to instantiate when array is indexed. If omitted then
                object is assumed to not be sliceable.

        Raises:
            StochasticallyDependentError:
                For dependency structures that can not later be
                rectified. This include under-defined
                distributions, and inclusion of distributions that
                should be exclusion.
        """
        assert isinstance(parameters, dict)
        self._parameters = parameters
        self._dependencies = list(dependencies)
        if rotation is None:
            rotation = sorted(enumerate(self._dependencies), key=lambda x: len(x[1]))
            rotation = [key for key, _ in rotation]
        rotation = list(rotation)
        assert len(set(rotation)) == len(dependencies)
        assert min(rotation) == 0
        assert max(rotation) == len(dependencies)-1
        self._rotation = rotation
        if exclusion is None:
            exclusion = set()
        self._exclusion = set(exclusion)
        if repr_args is None:
            repr_args = ("{}={}".format(key, self._parameters[key])
                         for key in sorted(self._parameters))
        self._repr_args = list(repr_args)
        self._mom_cache = {(0,)*len(self): 1.}
        self._ttr_cache = {}
        self._index_cls = index_cls
        self._indices = {}

        all_dependencies = {dep for deps in self._dependencies for dep in deps}
        if len(all_dependencies) < len(self):
            raise chaospy.StochasticallyDependentError(
                "%s is an under-defined probability distribution." % self)

        for key, param in parameters.items():
            if isinstance(param, Distribution):
                if all_dependencies.intersection(param._exclusion):
                    raise chaospy.StochasticallyDependentError((
                        "%s contains dependencies that can not also exist "
                        "other places in the dependency hierarchy") % param)
                self._exclusion.update(param._exclusion)
            else:
                self._parameters[key] = numpy.asarray(param)

    def _check_dependencies(self):
        """
        Check if the dependency structure is valid.

        Rosenblatt transformations, density calculations etc. assumes
        that the input and output of transformation is the same. It
        also assumes that there is a order defined in `self._rotation` so an
        decomposition `p(x0), p(x1|x0), ...` is possible.

        Raises:
            StochasticallyDependentError:
                If invalid dependency structure is present.
        """
        current = set()
        for idx in self._rotation:
            length = len(current)
            current.update(self._dependencies[idx])
            if len(current) != length+1:
                raise chaospy.StochasticallyDependentError(
                    "%s has more underlying dependencies than the size of distribution." % self)

    def _check_parameters(self, parameters):
        """
        Check if the parameters are as expected.

        Override in sub-class to add rules for how parameters should behave.
        """
        del parameters

    def get_parameters(self, cache, assert_numerical=True):
        """Get distribution parameters."""
        del assert_numerical
        out = self._parameters.copy()
        out["cache"] = cache
        return out

    def _declare_dependencies(self, count):
        """
        Declare stochastic dependency to an underlying random variable.

        Args:
            count (int):
                The number of variables to declare.

        Returns:
            (List[int]):
                Unique integer identifiers that represents dependencies.

        """
        length = len(DISTRIBUTION_IDENTIFIERS)
        new_identifiers = list(range(length+1, length+1+count))
        for idx in new_identifiers:
            DISTRIBUTION_IDENTIFIERS[idx] = self
        return new_identifiers

    @property
    def lower(self):
        """Lower bound for the distribution."""
        return self._get_lower(cache={})

    def _get_lower(self, cache):
        """In-processes function for getting lower bounds."""
        if self in cache:
            return cache[self][0]
        out = self._lower(**self.get_parameters(cache=cache, assert_numerical=False))
        assert not isinstance(out, Distribution), (self, out)
        out = numpy.atleast_1d(out)
        assert len(out) == len(self), (self, out)
        cache[self] = (out, None)
        return out

    def _lower(self, **kwargs):
        """Backend lower bound."""
        return self._ppf(numpy.array([1e-10]*len(self)), **kwargs)

    @property
    def upper(self):
        """Upper bound for the distribution."""
        cache = {}
        return self._get_upper(cache=cache)

    def _get_upper(self, cache):
        """In-processes function for getting upper bounds."""
        if self in cache:
            return cache[self][0]
        out = self._upper(**self.get_parameters(cache=cache, assert_numerical=False))
        assert not isinstance(out, Distribution), (self, out)
        out = numpy.atleast_1d(out)
        assert len(out) == len(self), (self, out)
        cache[self] = (out, None)
        return out

    def _upper(self, **kwargs):
        """Backend upper bound."""
        return self._ppf(numpy.array([1-1e-10]*len(self)), **kwargs)

    def fwd(self, x_data):
        """
        Forward Rosenblatt transformation.

        Args:
            x_data (numpy.ndarray):
                Location for the distribution function. ``x_data.shape`` must
                be compatible with distribution shape.

        Returns:
            (numpy.ndarray):
                Evaluated distribution function values, where
                ``out.shape==x_data.shape``.
        """
        logger = logging.getLogger(__name__)
        self._check_dependencies()
        x_data = numpy.asfarray(x_data)
        shape = x_data.shape
        x_data = x_data.reshape(len(self), -1)

        q_data = numpy.zeros(x_data.shape)
        indices = (x_data.T > self.upper.T).T
        q_data[indices] = 1
        indices = ~indices & (x_data.T >= self.lower).T
        if not numpy.all(indices):
            logger.debug("%s.fwd: %d/%d inputs out of bounds",
                         self, numpy.sum(~indices), len(indices))

        cache = {}
        q_data[indices] = self._get_fwd(x_data, cache)[indices]

        indices = (q_data > 1) | (q_data < 0)
        if numpy.any(indices):
            logger.debug("%s.fwd: %d/%d outputs out of bounds",
                         self, numpy.sum(indices), len(indices))
            q_data = numpy.clip(q_data, a_min=0, a_max=1)

        q_data = q_data.reshape(shape)
        return q_data

    def _get_fwd(self, x_data, cache):
        """In-process function for getting cdf-values."""
        if self in cache:
            return cache[self][1]
        assert len(x_data) == len(self), (
            "distribution %s is not of length %d" % (self, len(x_data)))
        lower = self._get_lower(cache=cache.copy())
        upper = self._get_upper(cache=cache.copy())
        parameters = self.get_parameters(cache=cache, assert_numerical=True)
        self._check_parameters(parameters)
        ret_val = self._cdf(x_data, **parameters)
        assert not isinstance(ret_val, Distribution), (self, ret_val)
        out = numpy.zeros(x_data.shape)
        out[:] = ret_val
        out[(x_data.T < lower.T).T] = 0
        out[(x_data.T > upper.T).T] = 1
        cache[self] = (x_data, out)
        return out

    @abc.abstractmethod
    def _cdf(self, *args, **kwargs):
        """Backend function for getting cdf-values."""
        pass

    def cdf(self, x_data):
        """
        Cumulative distribution function.

        Note that chaospy only supports cumulative distribution functions for
        stochastically independent distributions.

        Args:
            x_data (numpy.ndarray):
                Location for the distribution function. Assumes that
                ``len(x_data) == len(distribution)``.

        Returns:
            (numpy.ndarray):
                Evaluated distribution function values, where output has shape
                ``x_data.shape`` in one dimension and ``x_data.shape[1:]`` in
                higher dimensions.
        """
        self._check_dependencies()
        if self.stochastic_dependent:
            raise chaospy.StochasticallyDependentError(
                "Cumulative distribution does not support dependencies.")
        x_data = numpy.asarray(x_data)
        if self.interpret_as_integer:
            x_data = x_data+0.5
        q_data = self.fwd(x_data)
        if len(self) > 1:
            q_data = numpy.prod(q_data, 0)
        return q_data

    def inv(self, q_data, max_iterations=100, tollerance=1e-5):
        """
        Inverse Rosenblatt transformation.

        If possible the transformation is done analytically. If not possible,
        transformation is approximated using an algorithm that alternates
        between Newton-Raphson and binary search.

        Args:
            q_data (numpy.ndarray):
                Probabilities to be inverse. If any values are outside ``[0,
                1]``, error will be raised. ``q_data.shape`` must be compatible
                with distribution shape.
            max_iterations (int):
                If approximation is used, this sets the maximum number of
                allowed iterations in the Newton-Raphson algorithm.
            tollerance (float):
                If approximation is used, this set the error tolerance level
                required to define a sample as converged.

        Returns:
            (numpy.ndarray):
                Inverted probability values where
                ``out.shape == q_data.shape``.
        """
        logger = logging.getLogger(__name__)
        self._check_dependencies()
        q_data = numpy.asfarray(q_data)
        assert numpy.all((q_data >= 0) & (q_data <= 1)), "sanitize your inputs!"
        shape = q_data.shape
        q_data = q_data.reshape(len(self), -1)
        cache = {}
        x_data = self._get_inv(q_data, cache)

        indices = numpy.any((x_data.T < self.lower), axis=-1)
        if numpy.any(indices):
            logger.debug("%s.inv: %d/%d outputs under lower bound",
                         self, numpy.sum(indices), len(indices))
            x_data.T[indices] = self.lower

        indices = numpy.any((x_data.T > self.upper), axis=-1)
        if numpy.any(indices):
            logger.debug("%s.inv: %d/%d outputs over upper bound",
                         self, numpy.sum(indices), len(indices))
            x_data.T[indices] = self.upper

        x_data = x_data.reshape(shape)
        return x_data

    def _get_inv(self, q_data, cache):
        """In-process function for getting ppf-values."""
        if self in cache:
            return cache[self][0]
        if hasattr(self, "_ppf"):
            parameters = self.get_parameters(cache=cache, assert_numerical=True)
            self._check_parameters(parameters)
            ret_val = self._ppf(q_data, **parameters)
        else:
            ret_val = chaospy.approximate_inverse(self, q_data, cache)
        assert not isinstance(ret_val, Distribution), (self, ret_val)
        out = numpy.zeros(q_data.shape)
        out[:] = ret_val
        cache[self] = (out, q_data)
        return out

    def pdf(self, x_data, decompose=False):
        """
        Probability density function.

        If possible the density will be calculated analytically. If not
        possible, it will be approximated by approximating the one-dimensional
        derivative of the forward Rosenblatt transformation and multiplying the
        component parts. Note that even if the distribution is multivariate,
        each component of the Rosenblatt is one-dimensional.

        Args:
            x_data (numpy.ndarray):
                Location for the density function. If multivariate,
                `len(x_data) == len(self)` is required.
            decompose (bool):
                Decompose multivariate probability density `p(x), p(y|x), ...`
                instead of multiplying them together into `p(x, y, ...)`.

        Returns:
            (numpy.ndarray):
                Evaluated density function evaluated in `x_data`. If decompose,
                `output.shape == x_data.shape`, else if multivariate the first
                dimension is multiplied together.

        Example:
            >>> chaospy.Gamma(2).pdf([1, 2, 3, 4, 5]).round(3)
            array([0.368, 0.271, 0.149, 0.073, 0.034])
            >>> dist = chaospy.Iid(chaospy.Normal(0, 1), 2)
            >>> grid = numpy.mgrid[-1.5:2, -1.5:2]
            >>> dist.pdf(grid).round(3)
            array([[0.017, 0.046, 0.046, 0.017],
                   [0.046, 0.124, 0.124, 0.046],
                   [0.046, 0.124, 0.124, 0.046],
                   [0.017, 0.046, 0.046, 0.017]])
            >>> dist.pdf(grid, decompose=True).round(3)
            array([[[0.13 , 0.13 , 0.13 , 0.13 ],
                    [0.352, 0.352, 0.352, 0.352],
                    [0.352, 0.352, 0.352, 0.352],
                    [0.13 , 0.13 , 0.13 , 0.13 ]],
            <BLANKLINE>
                   [[0.13 , 0.352, 0.352, 0.13 ],
                    [0.13 , 0.352, 0.352, 0.13 ],
                    [0.13 , 0.352, 0.352, 0.13 ],
                    [0.13 , 0.352, 0.352, 0.13 ]]])

        """
        logger = logging.getLogger(__name__)
        self._check_dependencies()
        x_data = numpy.asfarray(x_data)
        shape = x_data.shape
        x_data = x_data.reshape(len(self), -1)

        f_data = numpy.zeros(x_data.shape)
        indices = numpy.any((x_data.T <= self.upper).T & (x_data.T >= self.lower).T, axis=0)
        if not numpy.all(indices):
            logger.debug("%s.fwd: %d/%d inputs out of bounds",
                         self, numpy.sum(~indices), len(indices))
        try:
            cache = {}
            f_data[:, indices] = self._get_pdf(x_data, cache)[:, indices]
        except chaospy.UnsupportedFeature:
            f_data[:, indices] = chaospy.approximate_density(self, x_data)[:, indices]

        f_data = f_data.reshape(shape)
        if len(self) > 1 and not decompose:
            f_data = numpy.prod(f_data, 0)
        return f_data

    def _get_pdf(self, x_data, cache):
        """In-process function for getting pdf-values."""
        if self in cache:
            return cache[self][1]
        lower = self._get_lower(cache=cache.copy())
        upper = self._get_upper(cache=cache.copy())
        index = numpy.all((x_data.T >= lower.T) & (x_data.T <= upper.T), axis=1)
        parameters = self.get_parameters(cache=cache, assert_numerical=True)
        self._check_parameters(parameters)
        out = numpy.zeros(x_data.shape)
        if hasattr(self, "_pdf"):
            ret_val = self._pdf(x_data, **parameters)
        else:
            raise chaospy.UnsupportedFeature(
                "%s: does not support analytical pdf." % self)
        _, ret_val = numpy.broadcast_arrays(out, ret_val)
        assert not isinstance(ret_val, Distribution), (self, ret_val)
        out[:, index] = ret_val[:, index]
        if self in cache:
            out = numpy.where(x_data == cache[self][0], out, 0)
        cache[self] = (x_data, out)
        return out

    def sample(self, size=(), rule="random", antithetic=None):
        """
        Create pseudo-random generated samples.

        By default, the samples are created using standard (pseudo-)random
        samples. However, if needed, the samples can also be created by either
        low-discrepancy sequences, and/or variance reduction techniques.

        Changing the sampling scheme, use the following ``rule`` flag:

        +----------------------+-----------------------------------------------+
        | key                  | Description                                   |
        +======================+===============================================+
        | ``chebyshev``        | Roots of first order Chebyshev polynomials.   |
        +----------------------+-----------------------------------------------+
        | ``nested_chebyshev`` | Chebyshev nodes adjusted to ensure nested.    |
        +----------------------+-----------------------------------------------+
        | ``korobov``          | Korobov lattice.                              |
        +----------------------+-----------------------------------------------+
        | ``random``           | Classical (Pseudo-)Random samples.            |
        +----------------------+-----------------------------------------------+
        | ``grid``             | Regular spaced grid.                          |
        +----------------------+-----------------------------------------------+
        | ``nested_grid``      | Nested regular spaced grid.                   |
        +----------------------+-----------------------------------------------+
        | ``latin_hypercube``  | Latin hypercube samples.                      |
        +----------------------+-----------------------------------------------+
        | ``sobol``            | Sobol low-discrepancy sequence.               |
        +----------------------+-----------------------------------------------+
        | ``halton``           | Halton low-discrepancy sequence.              |
        +----------------------+-----------------------------------------------+
        | ``hammersley``       | Hammersley low-discrepancy sequence.          |
        +----------------------+-----------------------------------------------+

        All samples are created on the ``[0, 1]``-hypercube, which then is
        mapped into the domain of the distribution using the inverse Rosenblatt
        transformation.

        Args:
            size (numpy.ndarray):
                The size of the samples to generate.
            rule (str):
                Indicator defining the sampling scheme.
            antithetic (bool, numpy.ndarray):
                If provided, will be used to setup antithetic variables. If
                array, defines the axes to mirror.

        Returns:
            (numpy.ndarray):
                Random samples with shape ``(len(self),)+self.shape``.
        """
        self._check_dependencies()
        size_ = numpy.prod(size, dtype=int)
        dim = len(self)
        if dim > 1:
            if isinstance(size, (tuple, list, numpy.ndarray)):
                shape = (dim,) + tuple(size)
            else:
                shape = (dim, size)
        else:
            shape = size

        from chaospy.distributions import sampler
        out = sampler.generator.generate_samples(
            order=size_, domain=self, rule=rule, antithetic=antithetic)
        try:
            out = out.reshape(shape)
        except:
            if len(self) == 1:
                out = out.flatten()
            else:
                out = out.reshape(dim, int(out.size/dim))

        if self.interpret_as_integer:
            out = numpy.round(out).astype(int)
        return out

    def mom(self, K, allow_approx=True, **kws):
        """
        Raw statistical moments.

        Creates non-centralized raw moments from the random variable. If
        analytical options can not be utilized, Monte Carlo integration
        will be used.

        Args:
            K (numpy.ndarray):
                Index of the raw moments. k.shape must be compatible with
                distribution shape.  Sampling scheme when performing Monte
                Carlo
            rule (str):
                rule for estimating the moment if the analytical method fails.
            antithetic (numpy.ndarray):
                List of bool. Represents the axes to mirror using antithetic
                variable during MCI.

        Returns:
            (numpy.ndarray):
                Shapes are related through the identity
                ``k.shape == dist.shape+k.shape``.
        """
        logger = logging.getLogger(__name__)
        K = numpy.asarray(K, dtype=int)
        assert numpy.all(K >= 0)
        shape = K.shape
        dim = len(self)

        if dim > 1:
            shape = shape[1:]

        size = int(K.size/dim)
        K = K.reshape(dim, size)
        try:
            out = [self._get_mom(kdata) for kdata in K.T]
            logger.debug("%s: PDF calculated successfully", str(self))
        except chaospy.UnsupportedFeature:
            if allow_approx:
                logger.info(
                    "%s: has stochastic dependencies; "
                    "Approximating moments with quadrature.", str(self))
                out = chaospy.approximate_moment(self, K)
            else:
                raise
        out = numpy.array(out)
        assert out.size == numpy.prod(shape), (out, shape)
        return out.reshape(shape)

    def _get_mom(self, kdata):
        """In-process function for getting moments."""
        if tuple(kdata) in self._mom_cache:
            return self._mom_cache[tuple(kdata)]
        parameters = self.get_parameters(cache={}, assert_numerical=False)
        ret_val = float(self._mom(kdata, **parameters))
        assert not isinstance(ret_val, Distribution), (self, ret_val)
        self._mom_cache[tuple(kdata)] = ret_val
        return ret_val

    def ttr(self, kloc):
        """
        Three terms relation's coefficient generator.

        Args:
            k (numpy.ndarray, int):
                The order of the coefficients.

        Returns:
            (Recurrence coefficients):
                Where out[0] is the first (A) and out[1] is the second
                coefficient With ``out.shape==(2,)+k.shape``.
        """
        self._check_dependencies()
        kloc = numpy.asarray(kloc, dtype=int)
        shape = kloc.shape
        kloc = kloc.reshape(len(self), -1)
        out = [self._get_ttr(k) for k in kloc.T]
        alpha, beta = numpy.asfarray(list(zip(*out)))
        out = numpy.array([alpha.T, beta.T])
        return out.reshape((2,)+shape)

    def _get_ttr(self, kdata):
        """In-process function for getting TTR-values."""
        if tuple(kdata) in self._ttr_cache:
            return self._ttr_cache[tuple(kdata)]
        parameters = self.get_parameters(cache={}, assert_numerical=True)
        alpha, beta = self._ttr(kdata, **parameters)
        assert not isinstance(alpha, Distribution), (self, alpha)
        assert not isinstance(beta, Distribution), (self, beta)
        alpha = numpy.asfarray(alpha).reshape(len(self))
        beta = numpy.asfarray(beta).reshape(len(self))
        self._ttr_cache[tuple(kdata)] = (alpha, beta)
        return alpha, beta

    def _get_cache_1(self, cache):
        """
        In-process function for getting cached values.

        Each time a distribution has been processed, the input and output
        values are stored in the cache.
        This checks if a distribution has been processed before and return a
        cache value if it is.

        The cached values are as follows:

        -----------  -------------
        Context      Content
        -----------  -------------
        pdf          Input values
        cdf/fwd      Input values
        ppf/inv      Output values
        lower/upper  Output values
        -----------  -------------

        Args:
            cache (Dict[Distribution, Tuple[numpy.ndarray, numpy.ndarray]]):
                Collection of cached values. Keys are distributions that has
                been processed earlier, values consist of up to two cache
                value.

        Returns:
            (numpy.ndarray, Distribution):
                The content of the first cache, if any. Else return self.
        """
        if self in cache:
            out = cache[self]
        else:
            parameters = self.get_parameters(cache=cache, assert_numerical=False)
            out = self._cache(**parameters)
        if not isinstance(out, Distribution):
            out = out[0]
        return out

    def _get_cache_2(self, cache):
        """
        In-process function for getting cached values.

        Each time a distribution has been processed, the input and output
        values are stored in the cache.
        This checks if a distribution has been processed before and return a
        cache value if it is.

        The cached values are as follows:

        -----------  -------------
        Context      Content
        -----------  -------------
        pdf          Output values
        cdf/fwd      Output values
        ppf/inv      Input values
        lower/upper  N/A
        -----------  -------------

        Args:
            cache (Dict[Distribution, Tuple[numpy.ndarray, numpy.ndarray]]):
                Collection of cached values. Keys are distributions that has
                been processed earlier, values consist of up to two cache
                value.

        Returns:
            (numpy.ndarray, Distribution):
                The content of the second cache, if any. Else return self.
        """
        if self in cache:
            out = cache[self]
        else:
            parameters = self.get_parameters(cache=cache, assert_numerical=False)
            out = self._cache(**parameters)
        if not isinstance(out, Distribution):
            out = out[1]
        return out

    @abc.abstractmethod
    def _cache(self, cache):
        """Backend function of retrieving cache values."""
        pass

    def __getitem__(self, index):
        if not hasattr(self, "_index_cls"):
            raise IndexError("indexing not supported")
        if isinstance(index, int):
            if not -len(self) < index < len(self):
                raise IndexError("index out of bounds: %s" % index)
            if index < 0:
                index += len(self)
            conditions = []
            for idx in self._rotation:
                if idx not in self._indices:
                    if conditions:
                        self._indices[idx] = self._index_cls(
                            parent=self, conditions=chaospy.J(*conditions))
                    else:
                        self._indices[idx] = self._index_cls(parent=self)
                if idx == index:
                    return self._indices[idx]
                conditions.append(self._indices[idx])
            return self._index_cls(self, chaospy.J(*conditions))
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = len(self) if index.stop is None else index.stop
            step = 1 if index.step is None else index.step
            return chaospy.J(*[self[idx] for idx in range(start, stop, step)])
        raise IndexError("unrecognized key")

    def __len__(self):
        """Distribution length."""
        return len(self._dependencies)

    def __repr__(self):
        """Distribution repr function."""
        args = ", ".join([str(arg) for arg in self._repr_args])
        return "{}({})".format(self.__class__.__name__, args)

    def __str__(self):
        """Distribution str function."""
        return repr(self)

    def __add__(self, X):
        """Y.__add__(X) <==> X+Y"""
        return chaospy.Add(self, X)

    def __radd__(self, X):
        """Y.__radd__(X) <==> Y+X"""
        return chaospy.Add(self, X)

    def __sub__(self, X):
        """Y.__sub__(X) <==> X-Y"""
        return chaospy.Add(self, -X)

    def __rsub__(self, X):
        """Y.__rsub__(X) <==> Y-X"""
        return chaospy.Add(X, -self)

    def __neg__(self):
        """X.__neg__() <==> -X"""
        return chaospy.Neg(self)

    def __mul__(self, X):
        """Y.__mul__(X) <==> X*Y"""
        return chaospy.Mul(self, X)

    def __rmul__(self, X):
        """Y.__rmul__(X) <==> Y*X"""
        return chaospy.Mul(X, self)

    def __div__(self, X):
        """Y.__div__(X) <==> Y/X"""
        return chaospy.Mul(self, X**-1)

    def __rdiv__(self, X):
        """Y.__rdiv__(X) <==> X/Y"""
        return chaospy.Mul(X, self**-1)

    def __floordiv__(self, X):
        """Y.__floordiv__(X) <==> Y/X"""
        return chaospy.Mul(self, X**-1)

    def __rfloordiv__(self, X):
        """Y.__rfloordiv__(X) <==> X/Y"""
        return chaospy.Mul(X, self**-1)

    def __truediv__(self, X):
        """Y.__truediv__(X) <==> Y/X"""
        return chaospy.Mul(self, X**-1)

    def __rtruediv__(self, X):
        """Y.__rtruediv__(X) <==> X/Y"""
        return chaospy.Mul(X, self**-1)

    def __pow__(self, X):
        """Y.__pow__(X) <==> Y**X"""
        return chaospy.Pow(self, X)

    def __rpow__(self, X):
        """Y.__rpow__(X) <==> X**Y"""
        return chaospy.Pow(X, self)

    # def __eq__(self, other):
    #     if not isinstance(other, Distribution):
    #         return False
    #     if len(other) != len(self):
    #         return False
    #     if len(other) > 1:
    #         if len(self) > 1:
    #             return all([d1 == d2 for d1, d2 in zip(self, other)])
    #         return all([self == d for d in other])
    #     elif len(self) > 1:
    #         return all([d == other for d in self])
    #     while isinstance(self, chaospy.J):
    #         self = self._parameters["_000"]
    #     while isinstance(other, chaospy.J):
    #         other = other._parameters["_000"]
    #     return self is other

    # def __hash__(self):
    #     return id(self)
