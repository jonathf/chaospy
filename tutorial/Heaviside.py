"""
Module for various functions related to the Heaviside function:

  * Exact and smoothed Heaviside function
  * Integral and derivative of exact and smoothed Heaviside function
  * Exact and smoothed indicator function
  * Integral and derivative of exact and smoothed indicator function
  * Exact and smoothed piecewise constant function
  * Integral and derivative of piecewise constant function

The exact functions are (usually) computed from direct considerations,
utilizing constantness or linearity in the formulas. The smoothed
versions are expressed in terms of the smoothed Heaviside function,
meaning that indicator functions are combinations of Heaviside functions,
and piecewise constant functions are combinations of indicator functions.
"""
import numpy as np
import math
from math import pi

class Heaviside:
    """
    Standard and smoothed Heaviside function.
    The function value of the standard/exact Heaviside function
    is 1 (int) for x greater than or equal to zero and 0 (int)
    otherwise. The smoothed version introduces a (small)
    interval $[-\epsilon, \epsilon]$ where the function behaves
    smoothly:
    $H(x) = 0.5 + \frac{x}{(2\epsilon} + \frac{1}\frac{2\pi}
    \sin(pi*x/\epsilon)$.
    """

    def __init__(self, eps=0):
        """`eps` is the smoothing parameter (0: exact Heaviside function."""
        self.eps = eps

    def __call__(self, x):
        if self.eps == 0:
            # Exact Heaviside function
            r = x >= 0
            if isinstance(x, (int,float)):
                return int(r)
            elif isinstance(x, np.ndarray):
                return np.asarray(r, dtype=int)
        else:
            # Smoothed Heaviside function
            if isinstance(x, (int,float)):
                return self._smooth_scalar(x)
            elif isinstance(x, np.ndarray):
                return self._smooth_vec(x)

    # Some different implementation of scalar and vectorized
    # Heaviside funcions with discontinuity
    def _exact_scalar(self, x):
        return 1 if x >= 0 else 0

    def _exact_bool(self, x):
        return x >= 0  # works for scalars and arrays, but returns bool

    def _exact_vec1(self, x):
        return np.where(x >= 0, 1, 0)

    def _exact_vec2(self, x):
        r = np.zeros_like(x)
        r[x >= 0] = 1
        return r

    # Implementations of smoothed Heaviside function
    def _smooth_scalar(self, x):
        eps = self.eps
        if x < -eps:
            return 0
        elif x > eps:
            return 1
        else:
            return 0.5 + x/(2*eps) + 1./(2*pi)*math.sin(pi*x/eps)

    def _smooth_vec(self, x):
        eps = self.eps
        r = np.zeros_like(x)
        condition1 = np.logical_and(x >= -eps, x <= eps)
        xc = x[condition1]
        #r[x <= 0] = 0.0  # not necessary since r has zeros
        r[condition1] = 0.5 + xc/(2*eps) + 1./(2*pi)*np.sin(pi*xc/eps)
        r[x > eps] = 1
        return r

    def plot(self, xmin=-1, xmax=1, center=0,
             resolution_outside=20, resolution_inside=200):
        """
        Return arrays x, y for plotting the Heaviside function
        H(x-`center`) on [`xmin`, `xmax`]. For the exact
        Heaviside function,
        ``x = [xmin, center, center, xmax]; y = [0, 0, 1, 1]``,
        while for the smoothed version, the ``x`` array
        is computed on basis of the `eps` parameter, with
        `resolution_outside` intervals on each side of the smoothed
        region and `resolution_inside` intervals in the smoothed region.
        """
        if self.eps == 0:
            return [xmin, center, center, xmax], [0, 0, 1, 1]
        else:
            n = float(resolution_inside)/self.eps
            x = np.concatenate((
                np.linspace(xmin, center-self.eps, resolution_outside+1),
                np.linspace(center-self.eps, center+self.eps, n+1),
                np.linspace(center+self.eps, xmax, resolution_outside+1)))
            y = self(x)
            return x, y

class IntegratedHeaviside:
    """
    The integral of the standard and smoothed Heaviside function,
    as represented by :class:`Heaviside`.
    """

    def __init__(self, eps=0):
        """
        `eps` is the smoothing parameter (0 corresponds to
        integrating the exact Heaviside function, otherwise
        the smoothed version.
        """
        self.eps = eps

    def __call__(self, x):
        if self.eps == 0:
            # Integral of exact Heaviside function
            r = x >= 0
            if isinstance(x, (int,float)):
                return float(x) if x >= 0 else 0.0
            elif isinstance(x, np.ndarray):
                r = np.zeros_like(x)
                r[x>=0] = x[x>=0]
                return r
            else:
                raise TypeError('x must be number or array, not %s' % type(x))
        else:
            # Integral of smoothed Heaviside function
            if isinstance(x, (int,float)):
                return self._smooth_scalar(x)
            elif isinstance(x, np.ndarray):
                return self._smooth_vec(x)
            else:
                raise TypeError('x must be number or array, not %s' % type(x))

    def _smooth_scalar(self, x):
        eps = self.eps
        if x < -eps:
            return 0
        elif x > eps:
            return x
        else:
            return -0.5*eps*math.cos(pi*x/eps)/pi**2 + \
                   0.5*x + x**2/(4*eps) - \
                   (0.5*eps/pi**2 - 0.5*eps + eps/4)

    def _smooth_vec(self, x):
        eps = self.eps
        r = np.zeros_like(x)
        condition1 = np.logical_and(x >= -eps, x <= eps)
        xc = x[condition1]
        #r[x <= 0] = 0.0  # not necessary since r has zeros
        r[condition1] = -0.5*eps*np.cos(pi*xc/eps)/pi**2 + 0.5*xc + \
                        xc**2/(4*eps) - (0.5*eps/pi**2 - 0.5*eps + eps/4)

        r[x > eps] = x[x > eps]
        return r

    def plot(self, xmin=-1, xmax=1, center=0,
             resolution_outside=20, resolution_inside=200):
        """
        Return arrays x, y for plotting the Heaviside function
        H(x-`center`) on [`xmin`, `xmax`]. For the exact
        Heaviside function,
        ``x = [xmin, center, xmax]; y = [0, 0, 1]``,
        while for the smoothed version, the ``x`` array
        is computed on basis of the `eps` parameter, with
        `resolution_outside` intervals on each side of the smoothed
        region and `resolution_inside` intervals in the smoothed region.
        """
        if self.eps == 0:
            return [xmin, center, xmax], [0, 0, xmax]
        else:
            n = float(resolution_inside)/self.eps
            x = np.concatenate((
                np.linspace(xmin, center-self.eps, resolution_outside+1),
                np.linspace(center-self.eps, center+self.eps, n+1),
                np.linspace(center+self.eps, xmax, resolution_outside+1)))
            y = self(x)
            return x, y


class DiracDelta:
    """
    The class represents a smoothed Dirac delta function:
    $\frac{1}{2\epsilon}(1 + \cos(\pi x/\epsilon)$ when
    $x\in [-\epsilon, \epsilon]$ and 0 elsewhere.
    DiracDelta is the derivative of the smoothed
    Heaviside function.
    """
    def __init__(self, eps, vectorized=False):
        self.eps = eps
        if self.eps == 0:
            raise ValueError('eps=0 is not allowed in class DiracDelta.')

    def __call__(self, x):
        if isinstance(x, (float, int)):
            return _smooth(x)
        elif isinstance(x, np.ndarray):
            return _smooth_vec(x)
        else:
            raise TypeError('x must be number or array, not %s' % type(x))

    def _smooth(self, x):
        eps = self.eps
        if x < -eps or x > eps:
            return 0
        else:
            return 1./(2*eps)*(1 + cos(pi*x/eps))

    def _smooth_vec(self, x):
        eps = self.eps
        r = np.zeros_like(x)
        condition1 - operator.and_(x >= -eps, x <= eps)
        xc = x[condition1]
        r[condition1] = 1./(2*eps)*(1 + np.cos(pi*xc/eps))
        return r

    def plot(self, center=0, xmin=-1, xmax=1,
             resolution_outside=20, resolution_inside=200):
        """
        Return arrays x, y for plotting the DiracDelta function
        centered in `center` on the interval [`xmin`, `xmax`].
        """
        n = float(resolution_inside)/self.eps
        x = np.concatenate((
            np.linspace(xmin, center-self.eps, resolution_outside+1),
            np.linspace(center-self.eps, center+self.eps,
                        resolution_inside+1),
            np.linspace(center+self.eps, xmax, resolution_outside+1)))
        y = self(x)
        return x, y

class IndicatorFunction:
    """
    Indicator function $I(x; L, R)$, which is 1 in $[L, R]$, and 0
    outside. Two parameters ``eps_L`` and ``eps_R`` can be set
    to provide smoothing of the left and right discontinuity
    in the indicator function. The indicator function is
    defined in terms of the Heaviside function (using class
    :class:`Heaviside`): $I(x; R, L) = H(x-L)H(R-x)$. All
    the smoothing is hence computed in ``Heaviside``.
    """
    def __init__(self, interval, eps_L=0, eps_R=0):
        """
        `interval` is a 2-tuple/list defining the interval [L, R] where
        the indicator function is 1.
        `eps` is a smoothing parameter: ``eps=0`` gives the standard
        discontinuous indicator function, while a value different
        from 0 gives rapid change from 0 to 1 over an interval of
        length 2*`eps`.
        """
        self.L, self.R = interval
        self.eps_L, self.eps_R = eps_L, eps_R
        self.Heaviside_L = Heaviside(eps_L)
        self.Heaviside_R = Heaviside(eps_R)

    def __call__(self, x):
        if self.eps_L == 0 and self.eps_R == 0:
            # Compute exact 0 and 1 values here rather than
            # using class Heaviside
            if isinstance(x, (float, int)):
                #return 0 if x < self.L or x >= self.R else 1
                return 0 if x < self.L or x > self.R else 1
            elif isinstance(x, np.ndarray):
                r = np.ones_like(x)
                r[x < self.L] = 0
                #r[x >= self.R] = 0
                r[x > self.R] = 0
                return r
        else:
            # Rely on class Heaviside for the smoothed version
            return self.Heaviside_L(x - self.L)*self.Heaviside_R(self.R - x)

    def plot(self, xmin=-1, xmax=1,
             resolution_outside=20, resolution_inside=200):
        """
        Return arrays x, y for plotting IndicatorFunction
        on [`xmin`, `xmax`]. For the exact discontinuous
        indicator function, we typically have
        ``x = [xmin, L, L, R, R, xmax]; y = [0, 0, 1, 1, 0, 0]``,
        while for the smoothed version, the densities of
        coordinates in the ``x`` array is computed on basis of the
        `eps` parameter with `resolution_outside` plotting intervals
        outside the smoothed regions and  `resolution_inside` intervals
        inside the smoothed regions.
        """
        if xmin > self.L or xmax < self.R:
            raise ValueError('xmin=%g > L=%g or xmax=%g < R=%g is meaningless for plot' % (xmin, self.L, xmax, self.R))

        if self.eps_L == 0 and self.eps_R == 0:
            return ([xmin, self.L, self.L, self.R, self.R, xmax],
                    [0, 0, 1, 1, 0, 0])
        else:
            n = float(resolution_inside)/(0.5*(self.eps_L + self.eps_R))
            x = np.concatenate((
                np.linspace(xmin, self.L-self.eps_L, resolution_outside+1),
                np.linspace(self.L-self.eps_L, self.R+self.eps_R,
                            resolution_inside+1),
                np.linspace(self.R+self.eps_R, xmax, resolution_outside+1)))
            y = self(x)
            return x, y

    def __str__(self):
        e = ', eps=%g' % self.eps if self.eps else ''
        return 'I(x)=1 on [%g, %g]%s' % (self.L, self.R, e)

    def __repr__(self):
        return 'IndicatorFunction([%g, %g], eps=%g)' % \
               (self.L, self.R, self.eps)


class IntegratedIndicatorFunction(IndicatorFunction):
    """
    Integral of the indicator functions as represented by
    :class:`IndicatorFunction`.
    """
    def __init__(self, interval, eps_L=0, eps_R=0):
        """
        `interval` is a 2-tuple/list defining the interval [L, R] where
        the indicator function is 1.
        `eps` is a smoothing parameter: ``eps=0`` gives the standard
        discontinuous indicator function, while a value different
        from 0 gives rapid change from 0 to 1 over an interval of
        length 2*`eps`.
        """
        self.L, self.R = interval
        self.eps_L, self.eps_R = eps_L, eps_R
        self.IntegratedHeaviside_L = IntegratedHeaviside(eps_L)
        self.IntegratedHeaviside_R = IntegratedHeaviside(eps_R)

    def __call__(self, x):
        if self.eps_L == 0 and self.eps_R == 0:
            if isinstance(x, (float, int)):
                if x < self.L:
                    return 0
                elif self.L <= x <= self.R:
                    return x - self.L
                else:
                    return self.R - self.R
            elif isinstance(x, np.ndarray):
                r = np.zeros_like(x)
                #r[x < self.L] = 0
                condition_middle = np.logical_and(x >= self.L, x <= self.R)
                r[condition_middle] = x[condition_middle] - self.L
                r[x > self.R] = self.R - self.L
                return r
            else:
                raise TypeError('x must be number or array, not %s' % type(x))
        else:
            # Integral of H(x-L)*H(R-x) can be approximated as
            # Integral of H(x-L) up to m=(L+R)/2 plus integral
            # of $H(R-x) for x>m. For x<=m, just integral of H(x-L).
            # All this assumes small eps...
            m = (self.L + self.R)/2.0  # middle point of interval
            if isinstance(x, (float, int)):
                if x < m:
                    return self.IntegratedHeaviside_L(x - self.L)
                else:
                    return self.IntegratedHeaviside_L(m - self.L) - \
                           self.IntegratedHeaviside_R(self.R - x) + \
                           self.IntegratedHeaviside_R(self.R - m)
            elif isinstance(x, np.ndarray):
                r = np.zeros_like(x)
                r[x < m] = self.IntegratedHeaviside_L(x - self.L)[x < m]
                r[x >= m] = self.IntegratedHeaviside_L(m - self.L) - \
                            self.IntegratedHeaviside_R(self.R - x)[x >= m] + \
                            self.IntegratedHeaviside_R(self.R - m)
                return r
            else:
                raise TypeError('x must be number or array, not %s' % type(x))

    def plot(self, xmin=-1, xmax=1,
             resolution_outside=20, resolution_inside=200):
        """
        Return arrays x, y for plotting IndicatorFunction
        on [`xmin`, `xmax`]. For the exact discontinuous
        indicator function, we typically have
        ``x = [xmin, L, R, xmax]; y = [0, 0, 1, 1]``,
        while for the smoothed version, the densities of
        coordinates in the ``x`` array is computed on basis of the
        `eps` parameter with `resolution_outside` plotting intervals
        outside the smoothed regions and  `resolution_inside` intervals
        inside the smoothed regions.
        """
        if xmin > self.L or xmax < self.R:
            raise ValueError('xmin=%g > L=%g or xmax=%g < R=%g is meaningless for plot' % (xmin, self.L, xmax, self.R))

        if self.eps_L == 0 and self.eps_R == 0:
            d = self.R - self.L
            return [xmin, self.L, self.R, xmax], [0, 0, d, d]
        else:
            n = float(resolution_inside)/(0.5*(self.eps_L + self.eps_R))
            x = np.concatenate((
                np.linspace(xmin, self.L-self.eps_L, resolution_outside+1),
                np.linspace(self.L-self.eps_L, self.R+self.eps_R,
                            resolution_inside+1),
                np.linspace(self.R+self.eps_R, xmax, resolution_outside+1)))
            y = self(x)
            return x, y

    def __str__(self):
        e = ', eps=%g' % self.eps if self.eps else ''
        return 'integral(I(x)=1 on [%g, %g]%s)' % (self.L, self.R, e)

    def __repr__(self):
        return 'IntegratedIndicatorFunction([%g, %g], eps=%g)' % \
               (self.L, self.R, self.eps)

class PiecewiseConstant:
    """
    Representation of a piecewise constant function.
    The discontinuities can be smoothed out.
    In this latter case the piecewise constant function is represented
    as a sum of indicator functions (:class:`IndicatorFunction`)
    times corresponding values.
    """
    def __init__(self, domain, data, eps=0):
        """
        `domain` is an interval (2-list, 2-tuple).
        `data[i][0]` holds the coordinate where the i-th
        interval starts, with `data[i][1]` as the corresponding
        function value. We have that ``data[0][0] = domain[0]``,
        and the last interval is ``[data[-1][0], domain[1]]``.
        `eps` is the smoothing factor: ``eps=0`` corresponds to
        a truly piecewise constant function.
        """
        self.L, self.R = domain
        self.data = data
        self.eps = eps
        if self.L != self.data[0][0]:
            raise ValueError('domain starts at %g, while data[0][0]=%g' % \
                             (self.L, self.data[0][0]))
        self._boundaries = [x for x, value in data]
        self._boundaries.append(self.R)
        self._values = [value for x, value in data]
        self._boundaries = np.array(self._boundaries, float)
        self._values = np.array(self._values, float)

        self._indicator_functions = []
        # Compute the indicator function on each interval i
        for i in range(len(self.data)):
            # Ensure eps_L=0 at the left and eps_R=0 at the right,
            # while both are eps at internal boundaries,
            # i.e., the function is always discontinuous at the start and end
            if i == 0:
                eps_L = 0; eps_R = eps  # left boundary
            elif i == len(self.data)-1:
                eps_R = 0; eps_L = eps  # right boundary
            else:
                eps_L = eps_R = eps     # internal boundary
            self._indicator_functions.append(IndicatorFunction(
                [self._boundaries[i], self._boundaries[i+1]],
                 eps_L=eps_L, eps_R=eps_R))

    def __call__(self, x):
        if self.eps == 0:
            return self.value(x)
        else:
            return sum(value*I(x) \
                       for I, value in \
                       zip(self._indicator_functions, self._values))

    def value(self, x):
        if isinstance(x, (float,int)):
            # Vectorized look up of the value corresponding to x:
            # lower_interval_limits = x >= self._boundaries[:-1]
            # x >= lower_interval_limits = [True, True, ..., False, ..., False]
            # self._values[x >= lower_interval_limits] = values for all
            # the intervals where x is to the right, the last one of
            # these is the right value corresponding to x.
            return self._values[x >= self._boundaries[:-1]][-1]
        elif isinstance(x, np.ndarray):
            a = np.array([self._values[xi >= self._boundaries[:-1]][-1]
                          for xi in x])
            return a
        else:
            raise TypeError('x must be number or array, not %s' % type(x))

    def plot(self,
             resolution_constant_regions=20,
             resolution_smooth_regions=200):
        """
        Return arrays x, y for plotting the piecewise constant function.
        Just the minimum number of straight lines are returned if
        ``eps=0``, otherwise `resolution_constant_regions` plotting intervals
        are insed in the constant regions with `resolution_smooth_regions`
        plotting intervals in the smoothed regions.
        """
        if self.eps == 0:
            x = []; y = []
            for I, value in zip(self._indicator_functions, self._values):
                x.append(I.L)
                y.append(value)
                x.append(I.R)
                y.append(value)
            return x, y
        else:
            n = float(resolution_smooth_regions)/self.eps
            if len(self.data) == 1:
                return [self.L, self.R], [self._values[0], self._values[0]]
            else:
                x = [np.linspace(self.data[0][0], self.data[1][0]-self.eps,
                                 resolution_constant_regions+1)]
                # Iterate over all internal discontinuities
                for I in self._indicator_functions[1:]:
                    x.append(np.linspace(I.L-self.eps, I.L+self.eps,
                                         resolution_smooth_regions+1))
                    x.append(np.linspace(I.L+self.eps, I.R-self.eps,
                                         resolution_constant_regions+1))
                # Last part
                x.append(np.linspace(I.R-self.eps, I.R, 3))
                x = np.concatenate(x)
                y = self(x)
                return x, y

class IntegratedPiecewiseConstant(PiecewiseConstant):
    """
    Representation of the integral of a piecewise constant function
    (as represented by :class:`PiecewiseConstant`).
    """
    def __init__(self, domain, data, eps=0):
        PiecewiseConstant.__init__(self, domain, data, eps)

        # Must recompute integrated indicator functions
        self._indicator_functions = []
        for i in range(len(self.data)):
            if i == 0:
                eps_L = 0; eps_R = eps  # left boundary
            elif i == len(self.data)-1:
                eps_R = 0; eps_L = eps  # right boundary
            else:
                eps_L = eps_R = eps     # internal boundary
            self._indicator_functions.append(
                IntegratedIndicatorFunction(
                    [self._boundaries[i], self._boundaries[i+1]],
                    eps_L=eps_L, eps_R=eps_R))

    def __call__(self, x):
        if self.eps == 0:
            return self.value(x)
        else:
            return sum(value*I(x) \
                       for I, value in \
                       zip(self._indicator_functions, self._values))

    def value(self, x):
        if isinstance(x, (float,int,np.float)):
            return self._value(x)
        elif isinstance(x, np.ndarray):
            return np.array([self._value(xi) for xi in x])
        else:
            raise TypeError('x must be number or array, not %s' % type(x))

    def _value(self, x):
        # See PiecewiseConstant.value for reasoning
        if isinstance(x, (float,int)):
            # Find index of interval no m where x is located
            v = self._values
            b = self._boundaries[:-1]
            m = len(v[x >= b]) - 1
            #part1 = np.cumsum((b[1:m+1] - b[:m])/a[:m]) # can be used for vectorization later
            part1 = np.sum((b[1:m+1] - b[:m])*v[:m])
            part2 = (x - b[m])*v[m]
            return part1 + part2
        elif isinstance(x, np.ndarray):
            raise NotImplementedError('vectorized version not implemented')
        else:
            raise TypeError('x must be number or array, not %s' % type(x))

    def plot(self,
             resolution_constant_regions=20,
             resolution_smooth_regions=200):
        """
        Return arrays x, y for plotting the piecewise constant function.
        Just the minimum number of straight lines are returned if
        ``eps=0``, otherwise `resolution_constant_regions` plotting intervals
        are insed in the constant regions with `resolution_smooth_regions`
        plotting intervals in the smoothed regions.
        """
        if self.eps == 0:
            x = []; y = []
            for b in self._boundaries:
                x.append(b)
                y.append(self(b))
            return np.array(x), np.array(y)
        else:
            n = float(resolution_smooth_regions)/self.eps
            if len(self.data) == 1:
                return [self.L, self.R], [self._values[0], self._values[0]]
            else:
                x = [np.linspace(self.data[0][0], self.data[1][0]-self.eps,
                                 resolution_constant_regions+1)]
                # Iterate over all internal discontinuities
                for I in self._indicator_functions[1:]:
                    x.append(np.linspace(I.L-self.eps, I.L+self.eps,
                                         resolution_smooth_regions+1))
                    x.append(np.linspace(I.L+self.eps, I.R-self.eps,
                                         resolution_constant_regions+1))
                # Last part
                x.append(np.linspace(I.R-self.eps, I.R, 3))
                x = np.concatenate(x)
                y = self(x)
                return x, y

import nose.tools as nt

def test_Heaviside():
    a = 3
    H = Heaviside()
    x = 3
    nt.assert_equal(H(x-a), 1)
    x = 2.95
    nt.assert_equal(H(x-a), 0)

    x = np.linspace(-0.5, 0.5, 3)
    exact = np.array([0, 1, 1], dtype=np.float)
    diff = np.abs(exact - H(x)).max()
    nt.assert_almost_equal(diff, 0, places=14)

def test_IntegratedHeaviside():
    a = 3
    IH = IntegratedHeaviside()
    x = 3
    nt.assert_equal(IH(x-a), 0)
    x = 4
    nt.assert_equal(IH(x-a), 1)

    x = np.linspace(-0.5, 2, 5)
    exact = np.array([ 0., 0.125, 0.75, 1.375, 2.])
    diff = np.abs(exact - IH(x)).max()
    nt.assert_almost_equal(diff, 0, places=14)

    # Smoothed version
    eps = 0.3
    IHs = IntegratedHeaviside(eps=eps)
    nt.assert_almost_equal(IHs(x[1]), 0.13138908, places=8)

    xmin = -1; xmax = 2
    IH_x, IH_y = IH.plot(xmin, xmax)
    diff = np.abs(np.array([-1, 0, 2]) - IH_x).max()
    nt.assert_almost_equal(diff, 0, places=14)
    diff = np.abs(np.array([ 0, 0, 2]) - IH_y).max()
    nt.assert_almost_equal(diff, 0, places=14)

    IHs_x, IHs_y = IHs.plot(xmin, xmax)
    IHs_y2 = IHs(IHs_x)  # testing __call__
    diff = np.abs(IHs_y2 - IHs_y).max()  # compare __call__ and plot
    nt.assert_almost_equal(diff, 0, places=14)

    return IH_x, IH_y, IHs_x, IHs_y, eps, xmin, xmax


def test_IndicatorFunction():
    data = [[0, 2], [2, 1], [4, 3]]

    # Exact indicator function
    L = data[1][0]
    R = data[2][0]
    I = IndicatorFunction([L, R], eps_L=0, eps_R=0)
    xmin = data[0][0]; xmax = data[-1][0] + 2
    I_x, I_y = I.plot(xmin, xmax)

    diff = np.abs(np.array([0, 2, 2, 4, 4, 6]) - I_x).max()
    nt.assert_equal(diff, 0)
    diff = np.abs(np.array([0, 0, 1, 1, 0, 0]) - I_y).max()
    nt.assert_equal(diff, 0)

    # Smoothed indicator function
    eps = 0.3
    Is = IndicatorFunction([L, R], eps_L=eps, eps_R=eps)
    Is_x, Is_y = Is.plot(xmin, xmax)
    Is_x_exact = np.array(
      [ 0.   ,  0.085,  0.17 ,  0.255,  0.34 ,  0.425,  0.51 ,  0.595,
        0.68 ,  0.765,  0.85 ,  0.935,  1.02 ,  1.105,  1.19 ,  1.275,
        1.36 ,  1.445,  1.53 ,  1.615,  1.7  ,  1.7  ,  1.713,  1.726,
        1.739,  1.752,  1.765,  1.778,  1.791,  1.804,  1.817,  1.83 ,
        1.843,  1.856,  1.869,  1.882,  1.895,  1.908,  1.921,  1.934,
        1.947,  1.96 ,  1.973,  1.986,  1.999,  2.012,  2.025,  2.038,
        2.051,  2.064,  2.077,  2.09 ,  2.103,  2.116,  2.129,  2.142,
        2.155,  2.168,  2.181,  2.194,  2.207,  2.22 ,  2.233,  2.246,
        2.259,  2.272,  2.285,  2.298,  2.311,  2.324,  2.337,  2.35 ,
        2.363,  2.376,  2.389,  2.402,  2.415,  2.428,  2.441,  2.454,
        2.467,  2.48 ,  2.493,  2.506,  2.519,  2.532,  2.545,  2.558,
        2.571,  2.584,  2.597,  2.61 ,  2.623,  2.636,  2.649,  2.662,
        2.675,  2.688,  2.701,  2.714,  2.727,  2.74 ,  2.753,  2.766,
        2.779,  2.792,  2.805,  2.818,  2.831,  2.844,  2.857,  2.87 ,
        2.883,  2.896,  2.909,  2.922,  2.935,  2.948,  2.961,  2.974,
        2.987,  3.   ,  3.013,  3.026,  3.039,  3.052,  3.065,  3.078,
        3.091,  3.104,  3.117,  3.13 ,  3.143,  3.156,  3.169,  3.182,
        3.195,  3.208,  3.221,  3.234,  3.247,  3.26 ,  3.273,  3.286,
        3.299,  3.312,  3.325,  3.338,  3.351,  3.364,  3.377,  3.39 ,
        3.403,  3.416,  3.429,  3.442,  3.455,  3.468,  3.481,  3.494,
        3.507,  3.52 ,  3.533,  3.546,  3.559,  3.572,  3.585,  3.598,
        3.611,  3.624,  3.637,  3.65 ,  3.663,  3.676,  3.689,  3.702,
        3.715,  3.728,  3.741,  3.754,  3.767,  3.78 ,  3.793,  3.806,
        3.819,  3.832,  3.845,  3.858,  3.871,  3.884,  3.897,  3.91 ,
        3.923,  3.936,  3.949,  3.962,  3.975,  3.988,  4.001,  4.014,
        4.027,  4.04 ,  4.053,  4.066,  4.079,  4.092,  4.105,  4.118,
        4.131,  4.144,  4.157,  4.17 ,  4.183,  4.196,  4.209,  4.222,
        4.235,  4.248,  4.261,  4.274,  4.287,  4.3  ,  4.3  ,  4.385,
        4.47 ,  4.555,  4.64 ,  4.725,  4.81 ,  4.895,  4.98 ,  5.065,
        5.15 ,  5.235,  5.32 ,  5.405,  5.49 ,  5.575,  5.66 ,  5.745,
        5.83 ,  5.915,  6.   ])
    Is_y_exact = np.array(
      [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   6.68624592e-05,   5.33414587e-04,
         1.79195007e-03,   4.22010747e-03,   8.17388231e-03,
         1.39810396e-02,   2.19350487e-02,   3.22896531e-02,
         4.52541748e-02,   6.09896409e-02,   7.96058045e-02,
         1.01159113e-01,   1.25651664e-01,   1.53031166e-01,
         1.83191907e-01,   2.15976711e-01,   2.51179843e-01,
         2.88550821e-01,   3.27799047e-01,   3.68599186e-01,
         4.10597186e-01,   4.53416831e-01,   4.96666697e-01,
         5.39947404e-01,   5.82858997e-01,   6.25008354e-01,
         6.66016458e-01,   7.05525406e-01,   7.43205041e-01,
         7.78759054e-01,   8.11930467e-01,   8.42506395e-01,
         8.70321977e-01,   8.95263431e-01,   9.17270160e-01,
         9.36335871e-01,   9.52508694e-01,   9.65890288e-01,
         9.76633962e-01,   9.84941839e-01,   9.91061115e-01,
         9.95279483e-01,   9.97919812e-01,   9.99334173e-01,
         9.99897318e-01,   9.99999756e-01,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   9.99999756e-01,   9.99897318e-01,
         9.99334173e-01,   9.97919812e-01,   9.95279483e-01,
         9.91061115e-01,   9.84941839e-01,   9.76633962e-01,
         9.65890288e-01,   9.52508694e-01,   9.36335871e-01,
         9.17270160e-01,   8.95263431e-01,   8.70321977e-01,
         8.42506395e-01,   8.11930467e-01,   7.78759054e-01,
         7.43205041e-01,   7.05525406e-01,   6.66016458e-01,
         6.25008354e-01,   5.82858997e-01,   5.39947404e-01,
         4.96666697e-01,   4.53416831e-01,   4.10597186e-01,
         3.68599186e-01,   3.27799047e-01,   2.88550821e-01,
         2.51179843e-01,   2.15976711e-01,   1.83191907e-01,
         1.53031166e-01,   1.25651664e-01,   1.01159113e-01,
         7.96058045e-02,   6.09896409e-02,   4.52541748e-02,
         3.22896531e-02,   2.19350487e-02,   1.39810396e-02,
         8.17388231e-03,   4.22010747e-03,   1.79195007e-03,
         5.33414587e-04,   6.68624592e-05,  -2.46510747e-17,
        -2.46510747e-17,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00])
    diff = np.abs(Is_x_exact - Is_x).max()
    nt.assert_almost_equal(diff, 0, places=14)
    diff = np.abs(Is_y_exact - Is_y).max()
    nt.assert_almost_equal(diff, 0, places=9)

    return I_x, I_y, Is_x, Is_y, eps, xmin, xmax

def test_IntegratedIndicatorFunction():
    data = [[0, 2], [2, 1], [4, 3]]

    # Exact integrated indicator function
    L = data[1][0]
    R = data[2][0]
    II = IntegratedIndicatorFunction([L, R], eps_L=0, eps_R=0)
    xmin = data[0][0]; xmax = data[-1][0] + 2
    II_x, II_y = II.plot(xmin, xmax)
    II_y2 = II(np.array(II_x, float))  # testing __call__
    diff = np.abs(II_y2 - II_y).max()  # compare __call__ and plot
    nt.assert_almost_equal(diff, 0, places=14)

    # Smoothed integrated indicator function
    eps = 0.3
    IIs = IntegratedIndicatorFunction([L, R], eps_L=eps, eps_R=eps)
    IIs_x, IIs_y = IIs.plot(xmin, xmax)#, 1, 4)

    IIs_y2 = IIs(IIs_x)  # testing __call__
    diff = np.abs(IIs_y2 - IIs_y).max()  # compare __call__ and plot
    nt.assert_almost_equal(diff, 0, places=14)

    return II_x, II_y, IIs_x, IIs_y, eps, xmin, xmax


def test_PiecewiseConstant():
    data = [[0, 2], [2, 1], [3, 3]]
    xmin = data[0][0]; xmax = data[-1][0] + 3

    # Exact piecewise constant function
    a = PiecewiseConstant([xmin, xmax], data, eps=0)
    a_x, a_y = a.plot()

    eps = 0.3
    as_ = PiecewiseConstant([xmin, xmax], data, eps=eps)
    as_x, as_y = as_.plot()

    a_x_exact = np.array([0.0, 2.0, 2.0, 3.0, 3.0, 6.0])
    diff = np.abs(a_x_exact - a_x).max()
    nt.assert_almost_equal(diff, 0, places=14)

    a_y_exact = np.array([2.0, 2.0, 1.0, 1.0, 3.0, 3.0])
    diff = np.abs(a_y_exact - a_y).max()
    nt.assert_almost_equal(diff, 0, places=14)

    as_x_exact = np.array(
      [ 0.   ,  0.085,  0.17 ,  0.255,  0.34 ,  0.425,  0.51 ,  0.595,
        0.68 ,  0.765,  0.85 ,  0.935,  1.02 ,  1.105,  1.19 ,  1.275,
        1.36 ,  1.445,  1.53 ,  1.615,  1.7  ,  1.7  ,  1.703,  1.706,
        1.709,  1.712,  1.715,  1.718,  1.721,  1.724,  1.727,  1.73 ,
        1.733,  1.736,  1.739,  1.742,  1.745,  1.748,  1.751,  1.754,
        1.757,  1.76 ,  1.763,  1.766,  1.769,  1.772,  1.775,  1.778,
        1.781,  1.784,  1.787,  1.79 ,  1.793,  1.796,  1.799,  1.802,
        1.805,  1.808,  1.811,  1.814,  1.817,  1.82 ,  1.823,  1.826,
        1.829,  1.832,  1.835,  1.838,  1.841,  1.844,  1.847,  1.85 ,
        1.853,  1.856,  1.859,  1.862,  1.865,  1.868,  1.871,  1.874,
        1.877,  1.88 ,  1.883,  1.886,  1.889,  1.892,  1.895,  1.898,
        1.901,  1.904,  1.907,  1.91 ,  1.913,  1.916,  1.919,  1.922,
        1.925,  1.928,  1.931,  1.934,  1.937,  1.94 ,  1.943,  1.946,
        1.949,  1.952,  1.955,  1.958,  1.961,  1.964,  1.967,  1.97 ,
        1.973,  1.976,  1.979,  1.982,  1.985,  1.988,  1.991,  1.994,
        1.997,  2.   ,  2.003,  2.006,  2.009,  2.012,  2.015,  2.018,
        2.021,  2.024,  2.027,  2.03 ,  2.033,  2.036,  2.039,  2.042,
        2.045,  2.048,  2.051,  2.054,  2.057,  2.06 ,  2.063,  2.066,
        2.069,  2.072,  2.075,  2.078,  2.081,  2.084,  2.087,  2.09 ,
        2.093,  2.096,  2.099,  2.102,  2.105,  2.108,  2.111,  2.114,
        2.117,  2.12 ,  2.123,  2.126,  2.129,  2.132,  2.135,  2.138,
        2.141,  2.144,  2.147,  2.15 ,  2.153,  2.156,  2.159,  2.162,
        2.165,  2.168,  2.171,  2.174,  2.177,  2.18 ,  2.183,  2.186,
        2.189,  2.192,  2.195,  2.198,  2.201,  2.204,  2.207,  2.21 ,
        2.213,  2.216,  2.219,  2.222,  2.225,  2.228,  2.231,  2.234,
        2.237,  2.24 ,  2.243,  2.246,  2.249,  2.252,  2.255,  2.258,
        2.261,  2.264,  2.267,  2.27 ,  2.273,  2.276,  2.279,  2.282,
        2.285,  2.288,  2.291,  2.294,  2.297,  2.3  ,  2.3  ,  2.32 ,
        2.34 ,  2.36 ,  2.38 ,  2.4  ,  2.42 ,  2.44 ,  2.46 ,  2.48 ,
        2.5  ,  2.52 ,  2.54 ,  2.56 ,  2.58 ,  2.6  ,  2.62 ,  2.64 ,
        2.66 ,  2.68 ,  2.7  ,  2.7  ,  2.703,  2.706,  2.709,  2.712,
        2.715,  2.718,  2.721,  2.724,  2.727,  2.73 ,  2.733,  2.736,
        2.739,  2.742,  2.745,  2.748,  2.751,  2.754,  2.757,  2.76 ,
        2.763,  2.766,  2.769,  2.772,  2.775,  2.778,  2.781,  2.784,
        2.787,  2.79 ,  2.793,  2.796,  2.799,  2.802,  2.805,  2.808,
        2.811,  2.814,  2.817,  2.82 ,  2.823,  2.826,  2.829,  2.832,
        2.835,  2.838,  2.841,  2.844,  2.847,  2.85 ,  2.853,  2.856,
        2.859,  2.862,  2.865,  2.868,  2.871,  2.874,  2.877,  2.88 ,
        2.883,  2.886,  2.889,  2.892,  2.895,  2.898,  2.901,  2.904,
        2.907,  2.91 ,  2.913,  2.916,  2.919,  2.922,  2.925,  2.928,
        2.931,  2.934,  2.937,  2.94 ,  2.943,  2.946,  2.949,  2.952,
        2.955,  2.958,  2.961,  2.964,  2.967,  2.97 ,  2.973,  2.976,
        2.979,  2.982,  2.985,  2.988,  2.991,  2.994,  2.997,  3.   ,
        3.003,  3.006,  3.009,  3.012,  3.015,  3.018,  3.021,  3.024,
        3.027,  3.03 ,  3.033,  3.036,  3.039,  3.042,  3.045,  3.048,
        3.051,  3.054,  3.057,  3.06 ,  3.063,  3.066,  3.069,  3.072,
        3.075,  3.078,  3.081,  3.084,  3.087,  3.09 ,  3.093,  3.096,
        3.099,  3.102,  3.105,  3.108,  3.111,  3.114,  3.117,  3.12 ,
        3.123,  3.126,  3.129,  3.132,  3.135,  3.138,  3.141,  3.144,
        3.147,  3.15 ,  3.153,  3.156,  3.159,  3.162,  3.165,  3.168,
        3.171,  3.174,  3.177,  3.18 ,  3.183,  3.186,  3.189,  3.192,
        3.195,  3.198,  3.201,  3.204,  3.207,  3.21 ,  3.213,  3.216,
        3.219,  3.222,  3.225,  3.228,  3.231,  3.234,  3.237,  3.24 ,
        3.243,  3.246,  3.249,  3.252,  3.255,  3.258,  3.261,  3.264,
        3.267,  3.27 ,  3.273,  3.276,  3.279,  3.282,  3.285,  3.288,
        3.291,  3.294,  3.297,  3.3  ,  3.3  ,  3.42 ,  3.54 ,  3.66 ,
        3.78 ,  3.9  ,  4.02 ,  4.14 ,  4.26 ,  4.38 ,  4.5  ,  4.62 ,
        4.74 ,  4.86 ,  4.98 ,  5.1  ,  5.22 ,  5.34 ,  5.46 ,  5.58 ,
        5.7  ,  5.7  ,  5.85 ,  6.   ])
    as_y_exact = np.array(
      [ 2.        ,  2.        ,  2.        ,  2.        ,  2.        ,
        2.        ,  2.        ,  2.        ,  2.        ,  2.        ,
        2.        ,  2.        ,  2.        ,  2.        ,  2.        ,
        2.        ,  2.        ,  2.        ,  2.        ,  2.        ,
        2.        ,  2.        ,  1.99999918,  1.99999342,  1.9999778 ,
        1.9999474 ,  1.99989732,  1.99982266,  1.99971858,  1.99958022,
        1.99940281,  1.99918158,  1.99891181,  1.99858884,  1.99820805,
        1.99776488,  1.99725483,  1.99667348,  1.99601646,  1.99527948,
        1.99445835,  1.99354893,  1.99254719,  1.99144918,  1.99025105,
        1.98894906,  1.98753954,  1.98601896,  1.98438388,  1.98263099,
        1.98075708,  1.97875905,  1.97663396,  1.97437896,  1.97199135,
        1.96946854,  1.96680809,  1.9640077 ,  1.96106519,  1.95797852,
        1.95474583,  1.95136535,  1.94783549,  1.9441548 ,  1.94032198,
        1.93633587,  1.93219548,  1.92789996,  1.92344861,  1.91884089,
        1.91407641,  1.90915494,  1.90407641,  1.89884089,  1.89344861,
        1.88789996,  1.88219548,  1.87633587,  1.87032198,  1.8641548 ,
        1.85783549,  1.85136535,  1.84474583,  1.83797852,  1.83106519,
        1.8240077 ,  1.81680809,  1.80946854,  1.80199135,  1.79437896,
        1.78663396,  1.77875905,  1.77075708,  1.76263099,  1.75438388,
        1.74601896,  1.73753954,  1.72894906,  1.72025105,  1.71144918,
        1.70254719,  1.69354893,  1.68445835,  1.67527948,  1.66601646,
        1.65667348,  1.64725483,  1.63776488,  1.62820805,  1.61858884,
        1.60891181,  1.59918158,  1.58940281,  1.57958022,  1.56971858,
        1.55982266,  1.54989732,  1.5399474 ,  1.5299778 ,  1.51999342,
        1.50999918,  1.5       ,  1.49000082,  1.48000658,  1.4700222 ,
        1.4600526 ,  1.45010268,  1.44017734,  1.43028142,  1.42041978,
        1.41059719,  1.40081842,  1.39108819,  1.38141116,  1.37179195,
        1.36223512,  1.35274517,  1.34332652,  1.33398354,  1.32472052,
        1.31554165,  1.30645107,  1.29745281,  1.28855082,  1.27974895,
        1.27105094,  1.26246046,  1.25398104,  1.24561612,  1.23736901,
        1.22924292,  1.22124095,  1.21336604,  1.20562104,  1.19800865,
        1.19053146,  1.18319191,  1.1759923 ,  1.16893481,  1.16202148,
        1.15525417,  1.14863465,  1.14216451,  1.1358452 ,  1.12967802,
        1.12366413,  1.11780452,  1.11210004,  1.10655139,  1.10115911,
        1.09592359,  1.09084506,  1.08592359,  1.08115911,  1.07655139,
        1.07210004,  1.06780452,  1.06366413,  1.05967802,  1.0558452 ,
        1.05216451,  1.04863465,  1.04525417,  1.04202148,  1.03893481,
        1.0359923 ,  1.03319191,  1.03053146,  1.02800865,  1.02562104,
        1.02336604,  1.02124095,  1.01924292,  1.01736901,  1.01561612,
        1.01398104,  1.01246046,  1.01105094,  1.00974895,  1.00855082,
        1.00745281,  1.00645107,  1.00554165,  1.00472052,  1.00398354,
        1.00332652,  1.00274517,  1.00223512,  1.00179195,  1.00141116,
        1.00108819,  1.00081842,  1.00059719,  1.00041978,  1.00028142,
        1.00017734,  1.00010268,  1.0000526 ,  1.0000222 ,  1.00000658,
        1.00000082,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.00000164,
        1.00001316,  1.00004439,  1.00010519,  1.00020536,  1.00035468,
        1.00056285,  1.00083955,  1.00119437,  1.00163684,  1.00217637,
        1.00282232,  1.0035839 ,  1.00447024,  1.00549034,  1.00665304,
        1.00796708,  1.00944103,  1.0110833 ,  1.01290214,  1.01490563,
        1.01710164,  1.0194979 ,  1.02210189,  1.02492092,  1.02796208,
        1.03123223,  1.03473802,  1.03848585,  1.04248189,  1.04673208,
        1.05124207,  1.0560173 ,  1.06106292,  1.06638381,  1.0719846 ,
        1.07786963,  1.08404295,  1.09050835,  1.09726931,  1.10432903,
        1.1116904 ,  1.11935605,  1.12732826,  1.13560904,  1.14420008,
        1.15310278,  1.16231823,  1.17184718,  1.18169011,  1.19184718,
        1.20231823,  1.21310278,  1.22420008,  1.23560904,  1.24732826,
        1.25935605,  1.2716904 ,  1.28432903,  1.29726931,  1.31050835,
        1.32404295,  1.33786963,  1.3519846 ,  1.36638381,  1.38106292,
        1.3960173 ,  1.41124207,  1.42673208,  1.44248189,  1.45848585,
        1.47473802,  1.49123223,  1.50796208,  1.52492092,  1.54210189,
        1.5594979 ,  1.57710164,  1.59490563,  1.61290214,  1.6310833 ,
        1.64944103,  1.66796708,  1.68665304,  1.70549034,  1.72447024,
        1.7435839 ,  1.76282232,  1.78217637,  1.80163684,  1.82119437,
        1.84083955,  1.86056285,  1.88035468,  1.90020536,  1.92010519,
        1.94004439,  1.96001316,  1.98000164,  2.        ,  2.01999836,
        2.03998684,  2.05995561,  2.07989481,  2.09979464,  2.11964532,
        2.13943715,  2.15916045,  2.17880563,  2.19836316,  2.21782363,
        2.23717768,  2.2564161 ,  2.27552976,  2.29450966,  2.31334696,
        2.33203292,  2.35055897,  2.3689167 ,  2.38709786,  2.40509437,
        2.42289836,  2.4405021 ,  2.45789811,  2.47507908,  2.49203792,
        2.50876777,  2.52526198,  2.54151415,  2.55751811,  2.57326792,
        2.58875793,  2.6039827 ,  2.61893708,  2.63361619,  2.6480154 ,
        2.66213037,  2.67595705,  2.68949165,  2.70273069,  2.71567097,
        2.7283096 ,  2.74064395,  2.75267174,  2.76439096,  2.77579992,
        2.78689722,  2.79768177,  2.80815282,  2.81830989,  2.82815282,
        2.83768177,  2.84689722,  2.85579992,  2.86439096,  2.87267174,
        2.88064395,  2.8883096 ,  2.89567097,  2.90273069,  2.90949165,
        2.91595705,  2.92213037,  2.9280154 ,  2.93361619,  2.93893708,
        2.9439827 ,  2.94875793,  2.95326792,  2.95751811,  2.96151415,
        2.96526198,  2.96876777,  2.97203792,  2.97507908,  2.97789811,
        2.9805021 ,  2.98289836,  2.98509437,  2.98709786,  2.9889167 ,
        2.99055897,  2.99203292,  2.99334696,  2.99450966,  2.99552976,
        2.9964161 ,  2.99717768,  2.99782363,  2.99836316,  2.99880563,
        2.99916045,  2.99943715,  2.99964532,  2.99979464,  2.99989481,
        2.99995561,  2.99998684,  2.99999836,  3.        ,  3.        ,
        3.        ,  3.        ,  3.        ,  3.        ,  3.        ,
        3.        ,  3.        ,  3.        ,  3.        ,  3.        ,
        3.        ,  3.        ,  3.        ,  3.        ,  3.        ,
        3.        ,  3.        ,  3.        ,  3.        ,  3.        ,
        3.        ,  3.        ,  3.        ])

    diff = np.abs(as_x_exact - as_x).max()
    nt.assert_almost_equal(diff, 0, places=14)

    diff = np.abs(as_y_exact - as_y).max()
    nt.assert_almost_equal(diff, 0, places=8)


    return a_x, a_y, as_x, as_y, eps, xmin, xmax

def test_IntegratedPiecewiseConstant():
    data = [[0, 2], [2, 1], [3, 3]]
    xmin = data[0][0]; xmax = data[-1][0] + 3

    # Integral of exact piecewise constant function
    A = IntegratedPiecewiseConstant([xmin, xmax], data, eps=0)
    A_x, A_y = A.plot()
    diff = np.abs(np.array([ 0.,  2.,  3.,  6.]) - A_x).max()
    nt.assert_almost_equal(diff, 0, places=14)
    diff = np.abs(np.array([  0.,   4.,   5.,  14.]) - A_y).max()
    nt.assert_almost_equal(diff, 0, places=8)

    # Integral of smoothed piecewise constant function
    eps = 0.3
    As = IntegratedPiecewiseConstant([xmin, xmax], data, eps=eps)
    As_x, As_y = As.plot(1, 2)

    diff = np.abs(As_x - np.array(
      [ 0.  ,  1.7 ,  1.7 ,  2.  ,  2.3 ,  2.3 ,  2.7 ,  2.7 ,  3.  ,
        3.3 ,  3.3 ,  5.7 ,  5.7 ,  5.85,  6.  ])).max()
    nt.assert_almost_equal(diff, 0, places=14)

    diff = np.abs(As_y - np.array(
      [  0.        ,   3.4       ,   3.4       ,   3.95539636,
         4.3       ,   4.3       ,   4.7       ,   4.7       ,
         5.08920729,   5.9       ,   5.9       ,  13.1       ,
        13.1       ,  13.55      ,  14.        ])).max()
    nt.assert_almost_equal(diff, 0, places=8)

    # Use higher resolution for plots
    As_x, As_y = As.plot(1, 6)
    return A_x, A_y, As_x, As_y, eps, xmin, xmax

def test_plot_IntegratedHeaviside(plot=False):
    IH_x, IH_y, IHs_x, IHs_y, eps, xmin, xmax = test_IntegratedHeaviside()
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(IH_x, IH_y)
        plt.hold('on')  # Matlab style
        plt.plot(IHs_x, IHs_y)
        plt.axis([xmin, xmax, -0.1, 3])
        #plt.legend(['integrated $H(x)$',
        #            'integrated $H(x; \epsilon=%g)$' % eps,])
        plt.title('Integrated Heaviside function')
        plt.show()

def test_plot_IndicatorFunction(plot=False):
    I_x, I_y, Is_x, Is_y, eps, xmin, xmax = test_IndicatorFunction()

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(I_x, I_y)
        plt.hold('on')  # Matlab style
        plt.plot(Is_x, Is_y)
        plt.axis([xmin, xmax, -0.1, 1.1])
        plt.legend(['$I(x)$', '$I(x; \epsilon=%g)$' % eps,])
        plt.title('Indicator function')
        plt.show()

def test_plot_IntegratedIndicatorFunction(plot=False):
    II_x, II_y, IIs_x, IIs_y, eps, xmin, xmax = \
          test_IntegratedIndicatorFunction()

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(II_x, II_y)
        plt.hold('on')  # Matlab style
        plt.plot(IIs_x, IIs_y)
        plt.axis([xmin, xmax, -1, 3])
        plt.legend(['integrated $I(x)$', 'integrated $I(x; \epsilon=%g)$' % eps,])
        plt.title('Integrated indicator function')
        plt.show()


def test_plot_PiecewiseConstant(plot=False):
    a_x, a_y, as_x, as_y, eps, xmin, xmax = test_PiecewiseConstant()
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(a_x, a_y)
        plt.plot(as_x, as_y)
        plt.legend(['$a(x)$',
                    '$a(x; \epsilon=%g)$' % eps,
                    ])
        plt.axis([xmin, xmax, -0.1, 4.5])
        plt.title('Piecewise constant function')
        plt.show()

def test_plot_IntegratedPiecewiseConstant(plot=False):
    A_x, A_y, As_x, As_y, eps, xmin, xmax = test_IntegratedPiecewiseConstant()
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(A_x, A_y)
        plt.plot(As_x, As_y)
        plt.legend(['$A(x)$',
                    '$A(x; \epsilon=%g)$' % eps,
                    ])
        plt.axis([xmin, xmax, -0.1, 15])
        plt.title('Integral of piecewise constant function')
        plt.show()

if __name__ == '__main__':
    import sys

    #test_Heaviside()
    #test_IntegratedHeaviside()
    #test_IndicatorFunction()
    #test_IntegratedIndicatorFunction()
    #test_PiecewiseConstant()
    #test_IntegratedPiecewiseConstant()
    test_plot_IntegratedHeaviside(plot=True)
    test_plot_IndicatorFunction(plot=True)
    test_plot_IntegratedIndicatorFunction(plot=True)
    test_plot_PiecewiseConstant(plot=True)
    test_plot_IntegratedPiecewiseConstant(plot=True)
