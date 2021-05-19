"""Abstract baseclass for all distributions."""
import logging
import numpy

import chaospy

from .utils import check_dependencies


class Distribution(object):
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

    def __init__(
            self,
            parameters,
            dependencies,
            rotation=None,
            exclusion=None,
            repr_args=None,
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
        self._mom_cache = {(0,)*len(dependencies): 1.}
        self._ttr_cache = {}
        self._indices = {}

        self._all_dependencies = {dep for deps in self._dependencies for dep in deps}
        if len(self._all_dependencies) < len(dependencies):
            raise chaospy.StochasticallyDependentError(
                "%s is an under-defined probability distribution." % self)

        for key, param in list(parameters.items()):
            if isinstance(param, Distribution):
                if self._all_dependencies.intersection(param._exclusion):
                    raise chaospy.StochasticallyDependentError((
                        "%s contains dependencies that can not also exist "
                        "other places in the dependency hierarchy") % param)
                self._exclusion.update(param._exclusion)
            else:
                self._parameters[key] = numpy.asarray(param)

    def get_parameters(self, idx, cache, assert_numerical=True):
        """Get distribution parameters."""
        del assert_numerical
        out = self._parameters.copy()
        assert isinstance(cache, dict)
        if idx is not None:
            assert not isinstance(idx, dict), idx
            assert idx == int(idx), idx
            assert "idx" not in out
        assert "cache" not in out
        out["cache"] = cache
        out["idx"] = idx
        return out

    @property
    def lower(self):
        """Lower bound for the distribution."""
        cache = {}
        out = numpy.zeros(len(self))
        for idx in self._rotation:
            out[idx] = self._get_lower(idx, cache=cache)
        return out

    def _get_lower(self, idx, cache):
        """In-processes function for getting lower bounds."""
        if (idx, self) in cache:
            return cache[idx, self][0]
        if hasattr(self, "get_lower_parameters"):
            parameters = self.get_lower_parameters(idx, cache)
        else:
            parameters = self.get_parameters(idx, cache, assert_numerical=False)
        out = self._lower(**parameters)
        assert not isinstance(out, Distribution), (self, out)
        out = numpy.atleast_1d(out)
        assert out.ndim == 1, (self, out, cache)
        cache[idx, self] = (out, None)
        return out

    def _lower(self, **kwargs):  # pragma: no cover
        """Backend lower bound."""
        raise chaospy.UnsupportedFeature("lower not supported")

    @property
    def upper(self):
        """Upper bound for the distribution."""
        cache = {}
        out = numpy.zeros(len(self))
        for idx in self._rotation:
            out[idx] = self._get_upper(idx, cache=cache)
        return out

    def _get_upper(self, idx, cache):
        """In-processes function for getting upper bounds."""
        if (idx, self) in cache:
            return cache[idx, self][0]
        if hasattr(self, "get_upper_parameters"):
            parameters = self.get_upper_parameters(idx, cache)
        else:
            parameters = self.get_parameters(idx, cache, assert_numerical=False)
        out = self._upper(**parameters)
        assert not isinstance(out, Distribution), (self, out)
        out = numpy.atleast_1d(out)
        assert out.ndim == 1, (self, out, cache)
        cache[idx, self] = (out, None)
        size = max([elem[0].size for elem in cache.values()])
        assert all([elem[0].size in (1, size) for elem in cache.values()])
        return out

    def _upper(self, **kwargs):  # pragma: no cover
        """Backend upper bound."""
        raise chaospy.UnsupportedFeature("lower not supported")

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
        check_dependencies(self)
        x_data = numpy.asfarray(x_data)
        shape = x_data.shape
        x_data = x_data.reshape(len(self), -1)
        cache = {}
        q_data = numpy.zeros(x_data.shape)
        for idx in self._rotation:
            q_data[idx] = self._get_fwd(x_data[idx], idx, cache)

        indices = (q_data > 1) | (q_data < 0)
        if numpy.any(indices):  # pragma: no cover
            logger.debug("%s.fwd: %d/%d outputs out of bounds",
                         self, numpy.sum(indices), len(indices))
            q_data = numpy.clip(q_data, a_min=0, a_max=1)

        q_data = q_data.reshape(shape)
        return q_data

    def _get_fwd(self, x_data, idx, cache):
        """In-process function for getting cdf-values."""
        logger = logging.getLogger(__name__)
        assert (idx, self) not in cache, "repeated evaluation"
        lower = numpy.broadcast_to(self._get_lower(idx, cache=cache.copy()), x_data.shape)
        upper = numpy.broadcast_to(self._get_upper(idx, cache=cache.copy()), x_data.shape)
        parameters = self.get_parameters(idx, cache, assert_numerical=True)
        ret_val = self._cdf(x_data, **parameters)
        assert not isinstance(ret_val, Distribution), (self, ret_val)
        out = numpy.zeros(x_data.shape)
        out[:] = ret_val
        indices = x_data < lower
        if numpy.any(indices):
            logger.debug("%s.fwd: %d/%d inputs below bounds",
                         self, numpy.sum(indices), len(indices))
            out = numpy.where(indices, 0, out)
        indices = x_data > upper
        if numpy.any(indices):
            logger.debug("%s.fwd: %d/%d inputs above bounds",
                         self, numpy.sum(indices), len(indices))
            out = numpy.where(indices, 1, out)
        assert numpy.all((out >= 0) | (out <= 1))

        cache[idx, self] = (x_data, out)
        assert out.ndim == 1, (self, out, cache)
        return out

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
        check_dependencies(self)
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
        check_dependencies(self)
        q_data = numpy.asfarray(q_data)
        assert numpy.all((q_data >= 0) & (q_data <= 1)), "sanitize your inputs!"
        shape = q_data.shape
        q_data = q_data.reshape(len(self), -1)
        cache = {}
        x_data = numpy.zeros(q_data.shape)
        for idx in self._rotation:
            x_data[idx] = self._get_inv(q_data[idx], idx, cache)

        x_data = x_data.reshape(shape)
        return x_data

    def _get_inv(self, q_data, idx, cache):
        """In-process function for getting ppf-values."""
        logger = logging.getLogger(__name__)
        assert numpy.all(q_data <= 1) and numpy.all(q_data >= 0)
        assert q_data.ndim == 1
        if (idx, self) in cache:
            return cache[idx, self][0]
        lower = numpy.broadcast_to(self._get_lower(idx, cache=cache.copy()), q_data.shape)
        upper = numpy.broadcast_to(self._get_upper(idx, cache=cache.copy()), q_data.shape)
        try:
            parameters = self.get_parameters(idx, cache, assert_numerical=True)
            ret_val = self._ppf(q_data, **parameters)
        except chaospy.UnsupportedFeature:
            ret_val = chaospy.approximate_inverse(
                self, idx, q_data, cache=cache)
        assert not isinstance(ret_val, Distribution), (self, ret_val)

        out = numpy.zeros(q_data.shape)
        out[:] = ret_val

        indices = out < lower
        if numpy.any(indices):
            logger.debug("%s.inv: %d/%d outputs below bounds",
                         self, numpy.sum(indices), len(indices))
            out = numpy.where(indices, lower, out)

        indices = out > upper
        if numpy.any(indices):
            logger.debug("%s.inv: %d/%d outputs above bounds",
                         self, numpy.sum(indices), len(indices))
            out = numpy.where(indices, upper, out)

        assert out.ndim == 1
        cache[idx, self] = (out, q_data)
        assert out.ndim == 1, (self, out, cache)
        return out

    def _ppf(self, xloc, **kwargs):
        raise chaospy.UnsupportedFeature(
            "%s: does not support analytical ppf." % self)

    def ppf(self, q_data, max_iterations=100, tollerance=1e-5):
        """
        Point percentile function.

        Also known as the inverse cumulative distribution function.

        Note that chaospy only supports point percentiles for univariate
        distributions.

        Args:
            q_data (numpy.ndarray):
                Probabilities to be inverse. If any values are outside ``[0,
                1]``, error will be raised.
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
        if len(self) > 1:
            raise ValueError(
                "only one-dimensional distribution supports percentiles.")
        return self.inv(
            q_data,
            max_iterations=max_iterations,
            tollerance=tollerance,
        )

    def pdf(self, x_data, decompose=False, allow_approx=True, step_size=1e-7):
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
            allow_approx (bool):
                Allow the density to be estimated using numerical derivative of
                forward mapping if analytical approach fails. Raises error
                instead if false.
            step_size (float):
                The relative step size between two points used to calculate the
                derivative, assuming approximation is being used.

        Raises:
            chaospy.UnsupportedFeature:
                If analytical calculation is not possible and `allow_approx` is
                false.

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
        check_dependencies(self)
        x_data = numpy.asfarray(x_data)
        shape = x_data.shape
        x_data = x_data.reshape(len(self), -1)
        f_data = numpy.zeros(x_data.shape)
        cache = {}
        for idx in self._rotation:
            try:
                cache_ = cache.copy()
                f_data[idx] = self._get_pdf(x_data[idx], idx, cache)
            except chaospy.UnsupportedFeature:
                if allow_approx:
                    logger.info(
                        "%s: has stochastic dependencies; "
                        "Approximating density with numerical derivative.", str(self)
                    )
                    cache = cache_
                    f_data[idx] = chaospy.approximate_density(
                        self, idx, x_data[idx], cache=cache, step_size=step_size)
                else:
                    raise

        f_data = f_data.reshape(shape)
        if len(self) > 1 and not decompose:
            f_data = numpy.prod(f_data, 0)
        return f_data

    def _get_pdf(self, x_data, idx, cache):
        """In-process function for getting pdf-values."""
        logger = logging.getLogger(__name__)
        assert x_data.ndim == 1
        if (idx, self) in cache:
            return cache[idx, self][1]
        lower = numpy.broadcast_to(self._get_lower(idx, cache=cache.copy()), x_data.shape)
        upper = numpy.broadcast_to(self._get_upper(idx, cache=cache.copy()), x_data.shape)
        parameters = self.get_parameters(idx, cache, assert_numerical=True)
        ret_val = self._pdf(x_data, **parameters)
        assert not isinstance(ret_val, Distribution), (self, ret_val)

        out = numpy.zeros(x_data.shape)
        out[:] = ret_val

        indices = (x_data < lower) | (x_data > upper)
        if numpy.any(indices):
            logger.debug("%s.fwd: %d/%d inputs out of bounds",
                         self, numpy.sum(indices), len(indices))
            logger.debug("%s[%s]: %s - %s - %s", self, idx, lower, x_data, upper)
            out = numpy.where(indices, 0, ret_val)

        if self in cache:
            out = numpy.where(x_data == cache[self][0], out, 0)
        cache[idx, self] = (x_data, out)
        assert out.ndim == 1, (self, out, cache)
        return out

    def _pdf(self, xloc, **kwargs):
        raise chaospy.UnsupportedFeature(
            "%s: does not support analytical pdf." % self)

    def sample(self, size=(), rule="random", antithetic=None, include_axis_dim=False, seed=None):
        """
        Create pseudo-random generated samples.

        By default, the samples are created using standard (pseudo-)random
        samples. However, if needed, the samples can also be created by either
        low-discrepancy sequences, and/or variance reduction techniques.

        Changing the sampling scheme, use the following ``rule`` flag:

        ----------------------  -------------------------------------------
        key                     description
        ----------------------  -------------------------------------------
        ``additive_recursion``  Modulus of golden ratio samples.
        ``chebyshev``           Roots of first order Chebyshev polynomials.
        ``grid``                Regular spaced grid.
        ``halton``              Halton low-discrepancy sequence.
        ``hammersley``          Hammersley low-discrepancy sequence.
        ``korobov``             Korobov lattice.
        ``latin_hypercube``     Latin hypercube samples.
        ``nested_chebyshev``    Chebyshev nodes adjusted to ensure nested.
        ``nested_grid``         Nested regular spaced grid.
        ``random``              Classical (Pseudo-)Random samples.
        ``sobol``               Sobol low-discrepancy sequence.
        ----------------------  -------------------------------------------

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
            include_axis_dim (bool):
                By default an extra dimension even if the number of dimensions
                is 1.
            seed (Optional[int]):
                If provided, fixes the random variable's seed, ensuring
                reproducible results.

        Returns:
            (numpy.ndarray):
                Random samples with ``self.shape``. An extra dimension might be
                added to the front if either ``len(dist) > 1`` or
                ``include_axis_dim=True``.

        """
        if seed is not None:
            state = numpy.random.get_state()
            numpy.random.seed(seed)
            out = self.sample(size, rule=rule, antithetic=antithetic,
                               include_axis_dim=include_axis_dim)
            numpy.random.set_state(state)
            return out

        check_dependencies(self)
        size_ = numpy.prod(size, dtype=int)
        dim = len(self)
        shape = ((size,) if isinstance(size, (int, float, numpy.number)) else tuple(size))
        shape = (-1,)+shape[1:]
        shape = shape if dim == 1 and not include_axis_dim else (dim,)+shape

        from chaospy.distributions import sampler
        out = sampler.generator.generate_samples(
            order=size_, domain=self, rule=rule, antithetic=antithetic)

        for idx, dist in enumerate(self):
            if dist.interpret_as_integer:
                out[idx] = numpy.round(out[idx])
        if self.interpret_as_integer:
            out = numpy.round(out).astype(int)
        out = out.reshape(shape)
        return out

    def mom(self, K, allow_approx=True, **kwargs):
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
            allow_approx (bool):
                Allow the moments to be calculated using quadrature integration
                if analytical approach fails. Raises error instead if false.
            kwargs (Any):
                Arguments passed to :func:`chaospy.approximate_moment` if
                approximation is used.

        Raises:
            chaospy.UnsupportedFeature:
                If analytical calculation is not possible and `allow_approx` is
                false.

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
            assert len(self) == shape[0]
            shape = shape[1:]

        size = int(K.size/dim)
        K = K.reshape(dim, size)
        try:
            out = [self._get_mom(kdata) for kdata in K.T]
            logger.debug("%s: moment calculated successfully", str(self))
        except chaospy.UnsupportedFeature:
            if allow_approx:
                logger.info(
                    "%s: has stochastic dependencies; "
                    "Approximating moments with quadrature.", str(self))
                out = [chaospy.approximate_moment(self, kdata) for kdata in K.T]
            else:
                out = [self._get_mom(kdata) for kdata in K.T]
        out = numpy.array(out)
        assert out.size == numpy.prod(shape), (out, shape)
        return out.reshape(shape)

    def _get_mom(self, kdata):
        """In-process function for getting moments."""
        if tuple(kdata) in self._mom_cache:
            return self._mom_cache[tuple(kdata)]
        if hasattr(self, "get_mom_parameters"):
            parameters = self.get_mom_parameters()
        else:
            parameters = self.get_parameters(idx=None, cache={}, assert_numerical=False)
        assert "idx" not in parameters, (self, parameters)
        ret_val = float(self._mom(kdata, **parameters))
        assert not isinstance(ret_val, Distribution), (self, ret_val)
        self._mom_cache[tuple(kdata)] = ret_val
        return ret_val

    def _mom(self, kloc, **kwargs):
        raise chaospy.UnsupportedFeature(
            "moments not supported for this distribution")

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
        check_dependencies(self)
        kloc = numpy.asarray(kloc, dtype=int)
        shape = kloc.shape
        kloc = kloc.reshape(len(self), -1)
        out = numpy.zeros((2,)+kloc.shape)
        for idy, kloc_ in enumerate(kloc.T):
            for idx in range(len(self)):
                out[:, idx, idy] = self._get_ttr(kloc_[idx], idx)
        return out.reshape((2,)+shape)

    def _get_ttr(self, kdata, idx):
        """In-process function for getting TTR-values."""
        if (idx, kdata) in self._ttr_cache:
            return self._ttr_cache[idx, kdata]
        if hasattr(self, "get_ttr_parameters"):
            parameters = self.get_ttr_parameters(idx)
        else:
            parameters = self.get_parameters(idx, cache={}, assert_numerical=True)
        alpha, beta = self._ttr(kdata, **parameters)
        assert not isinstance(alpha, Distribution), (self, alpha)
        assert not isinstance(beta, Distribution), (self, beta)
        alpha = numpy.asfarray(alpha).item()
        beta = numpy.asfarray(beta).item()
        self._ttr_cache[idx, kdata] = (alpha, beta)
        return alpha, beta

    def _ttr(self, kloc, **kwargs):
        raise chaospy.UnsupportedFeature(
            "three terms recursion not supported for this distribution")

    def _get_cache(self, idx, cache, get=None):
        """
        In-process function for getting cached values.

        Each time a distribution has been processed, the input and output
        values are stored in the cache.
        This checks if a distribution has been processed before and return a
        cache value if it is.

        The cached values are as follows:

        -----------  -------------  -------------
        Context      Get 0          Get 1
        -----------  -------------  -------------
        pdf          Input values   Output values
        cdf/fwd      Input values   Output values
        ppf/inv      Output values  Input values
        lower/upper  Output values  N/A
        -----------  -------------  -------------

        Args:
            idx (int):
                Which dimension to get cache from.
            cache (Dict[Distribution, Tuple[numpy.ndarray, numpy.ndarray]]):
                Collection of cached values. Keys are distributions that has
                been processed earlier, values consist of up to two cache
                value.
            get (int):
                Which cache to retrieve.

        Returns:
            (numpy.ndarray, Distribution):
                The content of the cache, if any. Else return self.

        """
        if (idx, self) in cache:
            assert get in (0, 1)
            out = cache[idx, self][get]
        else:
            out = self._cache(idx=idx, cache=cache, get=get)
        return out

    def _cache(self, idx, cache, get):
        """Backend function of retrieving cache values."""
        return self

    def __getitem__(self, index):
        if isinstance(index, numpy.number):
            assert index.dtype == int
            index = int(index)
        if isinstance(index, int):
            if not -len(self) < index < len(self):
                raise IndexError("index out of bounds: %s" % index)
            if index < 0:
                index += len(self)
            return chaospy.ItemDistribution(int(index), self)
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = len(self) if index.stop is None else index.stop
            step = 1 if index.step is None else index.step
            return chaospy.J(*[self[idx] for idx in range(start, stop, step)])
        raise IndexError("unrecognized key: %s" % repr(index))

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

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
        return chaospy.Negative(self)

    def __mul__(self, X):
        """Y.__mul__(X) <==> X*Y"""
        return chaospy.Multiply(self, X)

    def __rmul__(self, X):
        """Y.__rmul__(X) <==> Y*X"""
        return chaospy.Multiply(X, self)

    def __div__(self, X):
        """Y.__div__(X) <==> Y/X"""
        return chaospy.Multiply(self, X**-1)

    def __rdiv__(self, X):
        """Y.__rdiv__(X) <==> X/Y"""
        return chaospy.Multiply(X, self**-1)

    def __floordiv__(self, X):
        """Y.__floordiv__(X) <==> Y/X"""
        return chaospy.Multiply(self, X**-1)

    def __rfloordiv__(self, X):
        """Y.__rfloordiv__(X) <==> X/Y"""
        return chaospy.Multiply(X, self**-1)

    def __truediv__(self, X):
        """Y.__truediv__(X) <==> Y/X"""
        return chaospy.Multiply(self, X**-1)

    def __rtruediv__(self, X):
        """Y.__rtruediv__(X) <==> X/Y"""
        return chaospy.Multiply(X, self**-1)

    def __pow__(self, X):
        """Y.__pow__(X) <==> Y**X"""
        return chaospy.Power(self, X)

    def __rpow__(self, X):
        """Y.__rpow__(X) <==> X**Y"""
        return chaospy.Power(X, self)

    def __eq__(self, other):
        if not isinstance(other, Distribution):
            return False
        if len(other) != len(self):
            return False
        if len(self) > 1:
            return all([self == other for self, other in zip(self, other)])
        if isinstance(self, chaospy.ItemDistribution) and isinstance(other, chaospy.ItemDistribution):
            return (self._parameters["index"] == other._parameters["index"] and
                    self._parameters["parent"] is other._parameters["parent"])

        return self is other

    def __hash__(self):
        return id(self)
