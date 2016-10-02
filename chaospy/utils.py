"""
Common tools that does not fit anywhere else

Functions
---------
combine         Combine along dimension samples generated from
                different samplers
lazy_eval       Lazy evaluation decorator
load_lazy_eval  Load lazy_eval from disk
acf             Auto Correlation Function
pso             Particle Swarm optimization rutine
mlog10          Fast method for calculating the interpart of log10
mci             Monte Carlo integration tool
rlstsq          Least squares w/regularization
"""

import numpy as np
from numpy import random as r, linalg as la
from scipy.optimize import fmin
try:
    import pylab as pl
    import matplotlib.font_manager as fm
except:
    pl = None
    fm = None

from time import clock
import pickle
import os

import chaospy as cp

__all__ = [
    "combine",
    "lazy_eval",
    "lazy",
    "hashable",
    "acf",
    "pso",
    "mlog10",
    "mci",
    "rlstsq",
]


def combine(X, part=None):
    """
All linear combination of a list of list

Parameters
----------
X : array_like
    List of input arrays
    X[i] : array_like
        Components to take linear combination of with
        `X[i].shape=(N[i], M[i])` where N is to be taken linear
        combination of and M is static.
        M[i] is set to 1 if missing.


Returns
-------
Z : matrix of combinations with shape (np.prod(N), np.sum(M)).

Examples
--------
>>> A, B = [1,2], [[4,4],[5,6]]
>>> print(cp.combine([A, B]))
[[ 1.  4.  4.]
 [ 1.  5.  6.]
 [ 2.  4.  4.]
 [ 2.  5.  6.]]
"""
    def clean(x):
        x = np.array(x)
        if len(x.shape)<=1:
            x = x.reshape(x.size, 1)
        elif len(x.shape)>2:
            raise ValueError("shapes must be smaller than 3")
        return x

    X = [clean(x) for x in X]

    if not (part is None):
        parts, orders = part
        if np.array(orders).size==1:
            orders = [int(np.array(orders).item())]*len(X)
        parts = np.array(parts).flatten()

        for i in range(len(X)):
            m,n = float(parts[i]), float(orders[i])
            l = len(X[i])
            X[i] = X[i][int(m/n*l):int((m+1)/n*l)]

    shapes = [x.shape for x in X]
    size = np.prod(shapes, 0)[0]*np.sum(shapes, 0)[1]

    if size>10**9:
        raise MemoryError("Too large sets")

    if len(X)==1:
        out = X[0]
    elif len(X)==3:
        X1 = combine(X[:2])
        out = combine([X1,X[2]])
    elif len(X)>3:
        X1 = combine(X[:2])
        X2 = combine(X[2:])
        out = combine([X1,X2])
    else:
        x1,x2 = X
        l1,d1 = x1.shape
        l2,d2 = x2.shape
        out = np.empty((l1*l2, d1+d2))
        out[:,:d1] = np.tile(x1, l2).reshape(l1*l2, d1)
        out[:,d1:] = np.tile(x2.T, l1).reshape(d2, l1*l2).T

    return out


def combine_bac(X, chunk=2):

    L = len(X)

    if L>chunk:
        X1 = combine(X[:chunk], chunk)
        X2 = combine(X[chunk:], chunk)
        X = combine([X1,X2])
        return X

    def _clean(x):
        x = np.array(x)
        shape = x.shape
        if len(shape)<=1:
            x = x.reshape(x.size, 1)
        elif len(shape)>2:
            raise ValueError("shapes must be smaller than 3")
        return x
    X = map(_clean, X)

    grid = np.mgrid[[slice(0,len(_),1) for _ in X]]
    grid = grid.reshape(L, grid.size/L)

    X = [X[i][grid[i]] for i in range(L)]
    X = np.concatenate(X, 1)

    return X


def hashable(arg):

    if isinstance(arg, np.ndarray):
        return arg.tostring()
    elif isinstance(arg, list):
        return tuple(arg)
    if isinstance(arg, tuple):
        return tuple(map(hashable, arg))
    return arg


class lazy_eval(object):
    """
Lazy evaluation decorator.

In the case where evaluation is many, but many are like,
lazy_eval ensures that the function only get called once
per unique set of parameters.

To see the underlying function's documentation, run:
`help(foo.func)`.

Functions
---------
save        Save module to disk

Example
-------
>>> def foo(*x):
...     print("evaluating")
...     return x
>>> foo = lazy_eval(foo)
>>> print(foo(4,5))
evaluating
(4, 5)
>>> print(foo(4,5))
(4, 5)
>>> print(foo(5,6))
evaluating
(5, 6)

See also
--------
load_lazy_eval
    """

    def __init__(self, func, convert=hashable, load=None):
        """
func : callable
    Function to be easy evaluated.
convert : callable, optional
    Pre-preprocessing av input to make it hashable.
        """

        self.container = {}
        self.func = func
        self.convert = convert

        if not (load is None) and os.path.isfile(load):
            self.load(load)

    def __call__(self, *args, **kws):

        items = sorted(kws.items(), key=lambda x:x[0])
        key = args + tuple(items)

        if not (self.convert is None):
            key = tuple(self.convert(_) for _ in key)

        if key in self.container:
            return self.container[key]

        element = self.func(*args, **kws)
        self.container[key] = element
        return element

    def save(self, name):
        """
Save lazy_eval to dist

Parameters
----------
name : str
    absolute or relative path to file
        """

        f = open(name, "w")
        pickle.dump(self.container, f)
        f.close()

    def load(self, name):
        """
Load up lazy_eval object from disk

Parameters
----------
name : str
    absolute or relative path to file
        """
        f = open(name, "r")
        container = pickle.load(f)
        f.close()
        self.container.update(container)

    def __len__(self):
        """
Number of unique function evaluations
        """
        return len(self.container)


def lazy(convert, load=None):
    def lazy_decorator(func):
        return lazy_eval(func, convert, load)
    return lazy_wrapper



def acf(x, length=20):
    return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0,1] \
        for i in range(1, length)])


def pso(func, x0, span=10, particles=None, c1=1.5, c2=1.5,
        w=0.73, maxiter=1000, eps=0.0001, seed=None,
        retall=False, verbose=0, callback=None):
    """
Minimization method using the particle swarm consept.

Returns : x_opt, [f_opt, flag, iter, time]

    x_opt : array_like
        Optimal vector.
    f_opt : float
        Function value at optimal.
    flag : boolean
        True if the convergence criterion was fullfilled.
    iter : integer
        Number of iterations.
    time : float
        Time used to solve the problem.

Parameters
----------

func : function
    Function that is being minimized.
    It should take a vector as first argument.

x_0 : array_like
    Initial vector

c1 : float
    Factor of influence for personal best.
    The higher the value, the more the particles
    moves towards its personal best.

c2 : float
    Factor of influence for global best. The higher the value,
    the more the particles moves towards the global best.

eps : float
    Primary stop criterion if all particles fulfills:
        sum(max((xgbest - xpbest)**2, key=sum)**2) < eps
    xpbest, xgbest are respectivly personal and global best.

maxiter : integer
    Number of iterations each particle is allowed to travel.

particles : integer
    Number of particles in the swarm.
    Defaults to 2*len(x_0)+4

w : float
    Factor of influence for a particle velocity. The higher
    the value, the more a particle follows the same path.
    I.e. less turning around.

seed : object
    Fixed seed used in the random value sampler.

span : float
    The range around the starting vector 
    the particles are spread.

retall : boolean
    If True return meta-variables along with the optimal value.

Example
-------
>>> import numpy as np
>>> func = lambda x: np.sum(x*x)
>>> pso(func, np.ones(5), seed=1984)
array([-0.0039152 , -0.00224043, -0.00282827,\
  0.00943838,  0.00691279])
    """

    # Initializing values.
    t = -clock()
    r.seed(seed)
    if isinstance(x0, (float, int)):
        x0 = np.asfarray([x0])
    else:
        x0 = np.asfarray(x0)
    if particles==None:
        particles = 2*len(x0)+4
    dim = (particles,) + x0.shape
    converged = False
    iterations = maxiter
    if callback!=None:
        def foo(x):
            result = func(x)
            callback(x)
            return result
        func = foo

    # Initializing the model
    x = x0 + span * (2 * r.random_sample(dim) - 1)
    v = span / 10. * (2 * r.random_sample(dim) - 1)
    xpbest = x[:]
    fpbest = np.array([func(_) for _ in x])
    index = np.argmin(fpbest)
    xgbest = xpbest[index]

    # Starting the process.
    for iter in range(maxiter):

        # Update velocity and possition
        v = w*v + c1* r.random_sample(dim) *(xpbest-x) \
                + c2* r.random_sample(dim) *(xgbest-x)
        x += v

        # Update x_pbest and f(x_pbest)
        fx = np.array([func(_) for _ in x])
        improve = (fx < fpbest)
        xpbest = (xpbest.T * (improve==0) + x.T*improve).T
        fpbest = fpbest * (improve==0) + fx*improve

        # Update x_gbest and f(x_gbest)
        index = np.argmin(fpbest)
        xgbest = xpbest[index]
        fgbest = fpbest[index]

        # Test for convergence
        if sum(max((xgbest - xpbest)**2, key=sum)**2) < eps:
            converged = True
            iterations = iter+1
            break

        if verbose and iter % 10==0:
            print(fgbest)

    t += clock()

    if verbose:
        if converged:
            print('Minimization terminated successfully.',)
        else:
            print('Maximum iteration exited.',)
        print('''
x_opt:          %s
func(x_opt):    %g
iter:           %d
time (sec):     %g''' % (xgbest, fgbest, iterations, t))
    if retall:
        return xgbest, fgbest, converged, iterations, t
    return xgbest

def mlog10(a):
    """Interger part of log10. Works on large int variables."""

    out = 0
    while a>=10**100:
        out += 100
        a = a//10**100

    while a>=10**10:
        out += 10
        a = a//10**10

    while a>=10:
        out += 1
        a = a//10

    return out

mlog10 = lazy_eval(mlog10, lambda x: int(max(x.flatten())))


def mci(func, dist, samples=10**5, args=(), kws={}):
    """
Inner product estimated using Monte Carlo integration.
Good for high dim and high complexity.

Returns
-------
Q : float
    Approximate integral of `func`.

Parameters
----------
func : callable
    Function to be integrated.
dist : Dist
    The distribution to sample from.
samples : int
    Number of samples used in the integration.
args : tuple
    Additional positional arguments passed to func.
kws : dict
    Additional keyword arguments passed to func.

See Also
--------
quadrature  Integration approximation using Gaussian quadrature

Examples
--------
>>> cp.seed(1000)
>>> foo = lambda x:x**3
>>> dist = cp.Uniform()
>>> print(cp.mci(foo, dist=dist))
0.250923402627
    """
    x = dist.sample(size=(len(dist), samples))
    return sum(func(*x))/samples



#  # We need a special font for the code below.  It can be downloaded this way:
#  import os
#  import urllib2
#  from scipy import interpolate, signal
#  if not os.path.exists('Humor-Sans.ttf'):
#      fhandle = urllib2.urlopen('http://antiyawn.com/uploads/Humor-Sans.ttf')
#      open('Humor-Sans.ttf', 'wb').write(fhandle.read())
#  
#      
#  def xkcd_line(x, y, xlim=None, ylim=None,
#                mag=1.0, f1=30, f2=0.05, f3=15):
#      """
#      Mimic a hand-drawn line from (x, y) data
#  
#      Parameters
#      ----------
#      x, y : array_like
#          arrays to be modified
#      xlim, ylim : data range
#          the assumed plot range for the modification.  If not specified,
#          they will be guessed from the  data
#      mag : float
#          magnitude of distortions
#      f1, f2, f3 : int, float, int
#          filtering parameters.  f1 gives the size of the window, f2 gives
#          the high-frequency cutoff, f3 gives the size of the filter
#      
#      Returns
#      -------
#      x, y : ndarrays
#          The modified lines
#      """
#      x = np.asarray(x)
#      y = np.asarray(y)
#      
#      # get limits for rescaling
#      if xlim is None:
#          xlim = (x.min(), x.max())
#      if ylim is None:
#          ylim = (y.min(), y.max())
#  
#      if xlim[1] == xlim[0]:
#          xlim = ylim
#          
#      if ylim[1] == ylim[0]:
#          ylim = xlim
#  
#      # scale the data
#      x_scaled = (x - xlim[0]) * 1. / (xlim[1] - xlim[0])
#      y_scaled = (y - ylim[0]) * 1. / (ylim[1] - ylim[0])
#  
#      # compute the total distance along the path
#      dx = x_scaled[1:] - x_scaled[:-1]
#      dy = y_scaled[1:] - y_scaled[:-1]
#      dist_tot = np.sum(np.sqrt(dx * dx + dy * dy))
#  
#      # number of interpolated points is proportional to the distance
#      Nu = int(200 * dist_tot)
#      u = np.arange(-1, Nu + 1) * 1. / (Nu - 1)
#  
#      # interpolate curve at sampled points
#      k = min(3, len(x) - 1)
#      res = interpolate.splprep([x_scaled, y_scaled], s=0, k=k)
#      x_int, y_int = interpolate.splev(u, res[0]) 
#  
#      # we'll perturb perpendicular to the drawn line
#      dx = x_int[2:] - x_int[:-2]
#      dy = y_int[2:] - y_int[:-2]
#      dist = np.sqrt(dx * dx + dy * dy)
#  
#      # create a filtered perturbation
#      coeffs = mag * np.random.normal(0, 0.01, len(x_int) - 2)
#      b = signal.firwin(f1, f2 * dist_tot, window=('kaiser', f3))
#      response = signal.lfilter(b, 1, coeffs)
#  
#      x_int[1:-1] += response * dy / dist
#      y_int[1:-1] += response * dx / dist
#  
#      # un-scale data
#      x_int = x_int[1:-1] * (xlim[1] - xlim[0]) + xlim[0]
#      y_int = y_int[1:-1] * (ylim[1] - ylim[0]) + ylim[0]
#      
#      return x_int, y_int
#  
#  
#  def XKCDify(ax, mag=1.0,
#              f1=50, f2=0.01, f3=15,
#              bgcolor='w',
#              origo=None,
#              xaxis_arrow='+',
#              yaxis_arrow='+',
#              ax_extend=0.1,
#              expand_axes=False):
#      """Make axis look hand-drawn
#  
#      This adjusts all lines, text, legends, and axes in the figure to look
#      like xkcd plots.  Other plot elements are not modified.
#  
#      Parameters
#      ----------
#      ax : Axes instance
#          the axes to be modified.
#      mag : float
#          the magnitude of the distortion
#      f1, f2, f3 : int, float, int
#          filtering parameters.  f1 gives the size of the window, f2 gives
#          the high-frequency cutoff, f3 gives the size of the filter
#      origo : (float, float)
#          The locations to draw the x and y axes.  If not specified, they
#          will be drawn from the bottom left of the plot
#      xaxis_arrow, yaxis_arrow : str
#          where to draw arrows on the x/y axes.  Options are '+', '-', '+-', or ''
#      ax_extend : float
#          How far (fractionally) to extend the drawn axes beyond the original
#          axes limits
#      expand_axes : bool
#          if True, then expand axes to fill the figure (useful if there is only
#          a single axes in the figure)
#      """
#      # Get axes aspect
#      ext = ax.get_window_extent().extents
#      aspect = (ext[3] - ext[1]) / (ext[2] - ext[0])
#  
#      xlim = ax.get_xlim()
#      ylim = ax.get_ylim()
#  
#      xspan = xlim[1] - xlim[0]
#      yspan = ylim[1] - xlim[0]
#  
#      xax_lim = (xlim[0] - ax_extend * xspan,
#                 xlim[1] + ax_extend * xspan)
#      yax_lim = (ylim[0] - ax_extend * yspan,
#                 ylim[1] + ax_extend * yspan)
#  
#      if origo is None:
#          origo = (None, None)
#      xaxis_loc, yaxis_loc = origo
#  
#      if xaxis_loc is None:
#          xaxis_loc = ylim[0]
#  
#      if yaxis_loc is None:
#          yaxis_loc = xlim[0]
#  
#      # Draw axes
#      xaxis = pl.Line2D([xax_lim[0], xax_lim[1]], [xaxis_loc, xaxis_loc],
#                        linestyle='-', color='k')
#      yaxis = pl.Line2D([yaxis_loc, yaxis_loc], [yax_lim[0], yax_lim[1]],
#                        linestyle='-', color='k')
#  
#      # Label axes3, 0.5, 'hello', fontsize=14)
#      ax.text(xax_lim[1], xaxis_loc - 0.02 * yspan, ax.get_xlabel(),
#              fontsize=14, ha='right', va='top', rotation=12)
#      ax.text(yaxis_loc - 0.02 * xspan, yax_lim[1], ax.get_ylabel(),
#              fontsize=14, ha='right', va='top', rotation=78)
#      ax.set_xlabel('')
#      ax.set_ylabel('')
#  
#      # Add title
#      ax.text(0.5 * (xax_lim[1] + xax_lim[0]), yax_lim[1],
#              ax.get_title(),
#              ha='center', va='bottom', fontsize=16)
#      ax.set_title('')
#  
#      Nlines = len(ax.lines)
#      lines = [xaxis, yaxis] + [ax.lines.pop(0) for i in range(Nlines)]
#  
#      for line in lines:
#          x, y = line.get_data()
#  
#          x_int, y_int = xkcd_line(x, y, xlim, ylim,
#                                   mag, f1, f2, f3)
#  
#          # create foreground and background line
#          lw = line.get_linewidth()
#          line.set_linewidth(2 * lw)
#          line.set_data(x_int, y_int)
#  
#          # don't add background line for axes
#          if (line is not xaxis) and (line is not yaxis):
#              line_bg = pl.Line2D(x_int, y_int, color=bgcolor,
#                                  linewidth=8 * lw)
#  
#              ax.add_line(line_bg)
#          ax.add_line(line)
#  
#      # Draw arrow-heads at the end of axes lines
#      arr1 = 0.03 * np.array([-1, 0, -1])
#      arr2 = 0.02 * np.array([-1, 0, 1])
#  
#      arr1[::2] += np.random.normal(0, 0.005, 2)
#      arr2[::2] += np.random.normal(0, 0.005, 2)
#  
#      x, y = xaxis.get_data()
#      if '+' in str(xaxis_arrow):
#          ax.plot(x[-1] + arr1 * xspan * aspect,
#                  y[-1] + arr2 * yspan,
#                  color='k', lw=2)
#      if '-' in str(xaxis_arrow):
#          ax.plot(x[0] - arr1 * xspan * aspect,
#                  y[0] - arr2 * yspan,
#                  color='k', lw=2)
#  
#      x, y = yaxis.get_data()
#      if '+' in str(yaxis_arrow):
#          ax.plot(x[-1] + arr2 * xspan * aspect,
#                  y[-1] + arr1 * yspan,
#                  color='k', lw=2)
#      if '-' in str(yaxis_arrow):
#          ax.plot(x[0] - arr2 * xspan * aspect,
#                  y[0] - arr1 * yspan,
#                  color='k', lw=2)
#  
#      # Change all the fonts to humor-sans.
#      prop = fm.FontProperties(fname='Humor-Sans.ttf', size=16)
#      for text in ax.texts:
#          text.set_fontproperties(prop)
#      
#      # modify legend
#      leg = ax.get_legend()
#      if leg is not None:
#          leg.set_frame_on(False)
#          
#          for child in leg.get_children():
#              if isinstance(child, pl.Line2D):
#                  x, y = child.get_data()
#                  child.set_data(xkcd_line(x, y, mag=10, f1=100, f2=0.001))
#                  child.set_linewidth(2 * child.get_linewidth())
#              if isinstance(child, pl.Text):
#                  child.set_fontproperties(prop)
#      
#      # Set the axis limits
#      ax.set_xlim(xax_lim[0] - 0.1 * xspan,
#                  xax_lim[1] + 0.1 * xspan)
#      ax.set_ylim(yax_lim[0] - 0.1 * yspan,
#                  yax_lim[1] + 0.1 * yspan)
#  
#      # adjust the axes
#      ax.set_xticks([])
#      ax.set_yticks([])      
#  
#      if expand_axes:
#          ax.figure.set_facecolor(bgcolor)
#          ax.set_axis_off()
#          ax.set_position([0, 0, 1, 1])
#      
#      return ax

def rlstsq(A, b, order=0, alpha=None, cross=False):
    """
Least Squares Minimization using Tikhonov regularization, and
robust generalized cross-validation.

Parameters
----------
A : array_like, shape (M,N)
    "Coefficient" matrix.
b : array_like, shape (M,) or (M, K)
    Ordinate or "dependent variable" values. If `b` is
    two-dimensional, the least-squares solution is calculated for
    each of the `K` columns of `b`.
order : int, array_like
    If int, it is the order of Tikhonov regularization.
    If array_like, it will be used as regularization matrix.
alpha : float, optional
    Dampening parameter.
    If omited, it will be calculated from robust generalized
    cross-validation.
    """

    A = np.array(A)
    b = np.array(b)
    m,l = A.shape

    if cross:
        out = np.empty((m,l) + b.shape[1:])
        A_ = np.empty((m-1,l))
        b_ = np.empty((m-1,) + b.shape[1:])
        for i in range(m):
            A_[:i] = A[:i]
            A_[i:] = A[i+1:]
            b_[:i] = b[:i]
            b_[i:] = b[i+1:]
            out[i] = rlstsq(A_, b_, order, alpha, False)

        return np.median(out, 0)

    if order==0:
        L = np.eye(l)
    elif order==1:
        L = np.zeros((l-1,l))
        L[:,:-1] -= np.eye(l-1)
        L[:,1:] += np.eye(l-1)
    elif order==2:
        L = np.zeros((l-2,l))
        L[:,:-2] += np.eye(l-2)
        L[:,1:-1] -= 2*np.eye(l-2)
        L[:,2:] += np.eye(l-2)
    elif order is None:
        L = np.zeros(1)
    else:
        L = np.array(order)
        assert L.shape[-1]==l or L.shape in ((), (1,))

    if alpha is None and not (order is None):

        gamma = .1

        def rgcv_error(alpha):
            A_ = np.dot(A.T,A)+alpha**2*(np.dot(L.T,L))
            A_ = np.dot(la.inv(A_), A.T)
            x = np.dot(A_, b)
            res2 = np.sum((np.dot(A,x)-b)**2)
            K = np.dot(A, A_)
            V = m*res2/np.trace(np.eye(m)-K)**2
            mu2 = np.trace(np.dot(K,K))/m

            return (gamma + (1-gamma)*mu2)*V

        alpha = np.abs(fmin(rgcv_error, 0., disp=0)[0])

    out = la.inv(np.dot(A.T,A) + alpha**2*np.dot(L.T, L))
    out = np.dot(out, np.dot(A.T, b))
    return out


if __name__=="__main__":
    import chaospy as cp
    import doctest
    doctest.testmod()
