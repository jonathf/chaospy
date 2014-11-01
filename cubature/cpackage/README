== Cubature Package ==

Multidimensional integration ("cubature") code by Steven G. Johnson
<stevenj@alum.mit.edu>, based in part on code from the GNU Scientific
Library (GSL) by Brian Gough and others and from the HIntLib
numeric-integration library by Rudolf Schuerer.  Adaptive integration
of either one integrand or a vector of integrands is supported.  Both
h-adaptive integration (partitioning the integration domain) and
p-adaptive integration (increasing the order of the integration rule)
are supported (the latter being often better for smooth functions in
low dimensions).

=== Download and Copyright ===

The latest version can be downloaded from: http://ab-initio.mit.edu/cubature/

  Copyright (c) 2005-2013 Steven G. Johnson
 
  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.
 
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
 
  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 
  Portions (see comments) based on HIntLib (also distributed under
  the GNU GPL, v2 or later), copyright (c) 2002-2005 Rudolf Schuerer.
       (http://www.cosy.sbg.ac.at/~rschuer/hintlib/)
  
  Portions (see comments) based on GNU GSL (also distributed under
  the GNU GPL, v2 or later), copyright (c) 1996-2000 Brian Gough.
       (http://www.gnu.org/software/gsl/)

=== h-adaptive cubature ===

The basic algorithm is based on the adaptive cubature described in
 
     A. C. Genz and A. A. Malik, "An adaptive algorithm for numeric
     integration over an N-dimensional rectangular region,"
     J. Comput. Appl. Math. 6 (4), 295-302 (1980).

and subsequently extended to integrating a vector of integrands in

     J. Berntsen, T. O. Espelid, and A. Genz, "An adaptive algorithm
     for the approximate calculation of multiple integrals,"
     ACM Trans. Math. Soft. 17 (4), 437-451 (1991).

Note, however, that we do not use any of code from the above authors
(in part because their code is Fortran 77, but mostly because it is
under the restrictive ACM copyright license).  I did make use of some
GPL code from Rudolf Schuerer's HIntLib and from the GNU Scientific
Library as noted above, however.

I am also grateful to Dmitry Turbiner <dturbiner@alum.mit.edu>, who
implemented an initial prototype of the "vectorized" functionality for
evaluating multiple points in a single call (as opposed to multiple
functions in a single call).  (Although Dmitry implemented a working
version, I ended up re-implementing this feature from scratch as part
of a larger code-cleanup, and in order to have a single code path for
the vectorized and non-vectorized APIs.)

I subsequently extended the "vectorized" interface to use an algorithm
by Gladwell to increase the potential parallelism:

     I. Gladwell, "Vectorization of one dimensional quadrature codes,"
     pp. 230--238 in _Numerical Integration. Recent Developments,
     Software and Applications_, G. Fairweather and P. M. Keast, eds.,
     NATO ASI Series C203, Dordrecht (1987).

as described in:

     J. M. Bull and T. L. Freeman, "Parallel globally adaptive
     algorithms for multi-dimensional integration,"
     http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.6638
     (1994).

=== p-adaptive cubature ===

The p-adaptive cubature routine uses a tensor product of
Clenshaw-Curtis quadrature rules (using the embedded half-order rules
for error estimation).  The degree of the rules is doubled along each
dimension (one dimension at a time) until convergence is achieved.

This is often superior to h-adaptive cubature for smooth (especially
analytic) integrands in low dimensions (particularly 1 or 2 dimensions),
especially when high accuracy is needed.  It is not usually a good idea
in more than 3 dimensions.

The quadrature weights and points are precomputed with a program
clencurt_gen.c and stored in a file clencurt.h.  The default
clencurt.h file includes Clenshaw-Curtis rules up to order 2^20 (about
10^6) along each dimension, which should normally suffice for any
smooth function that is not extremely oscillatory.  (If not, pcubature
will return a nonzero result, indicating failure.)  If you need
higher-order rules, you can either switch to the h-adaptive routine or
recompile clencurt_gen (which requires FFTW, from www.fftw.org) and
run it with a larger argument m > 19 (corresponding to degree 2^(m+1)
rules).

-----------------------------------------------------------------------
== Usage ==

You should compile either hcubature.c or pcubature.c (or both),
depending on whether you are using h-adaptive or p-adaptive cubature
(or both).  Then link it with your program and #include the header
file cubature.h.

The central subroutine you will be calling is probably:

int hcubature(unsigned fdim, integrand f, void *fdata,
              unsigned dim, const double *xmin, const double *xmax,
              unsigned maxEval, double reqAbsError, double reqRelError,
              error_norm norm, double *val, double *err);

for h-adaptive cubature, or pcubature (same arguments) for
p-adaptive cubature.  (See also the vectorized interface below.)

This integrates a function F(x) returning a vector of FDIM
integrands, where x is a DIM-dimensional vector ranging from XMIN to
XMAX (i.e. in a hypercube XMIN[i] <= x[i] <= XMAX[i]).

MAXEVAL specifies a maximum number of function evaluations (0 for no
limit).  Otherwise, the integration stops when the estimated |error|
is less than REQABSERROR (the absolute error requested), or when the
estimated |error| is less than REQRELERROR * |integral value| (the
relative error requested).

For vector-valued integrands (FDIM > 1), NORM specifies the norm that
is used to measure the error and determine convergence properties.
(The NORM argument is irrelevant for FDIM<=1 and is ignored.)  Given
vectors v and e of estimated integrals and errors therein,
respectively, the NORM argument takes on one of the following
enumerated constant values:

* ERROR_L1, ERROR_L2, ERROR_LINF: the absolute error is measured
as |e| and the relative error as |e|/|v|, where |...| is the L1,
L2, or L-infinity norm, respectively.  (|x| in the L1 norm is the
sum of the absolute values of the components, in the L2 norm is the
root mean square of the components, and in the L-infinity norm is
the maximum absolute value of the components)

* ERROR_INDIVIDUAL: Convergence is achieved only when each integrand
(each component of v and e) individually satisfies the requested
error tolerances.

* ERROR_PAIRED: Like ERROR_INDIVIDUAL, except that the integrands
are grouped into consecutive pairs, with the error tolerance applied
in an L2 sense to each pair.  This option is mainly useful for
integrating vectors of complex numbers, where each consecutive pair
of real integrands is the real and imaginary parts of a single
complex integrand, and you only care about the error in the complex
plane rather than the error in the real and imaginary parts separately.

VAL and ERR are arrays of length FDIM, which upon return are the
computed integral values and estimated errors, respectively.  (The
estimated errors are based on an embedded cubature rule of lower
order; for smooth functions, this estimate is usually conservative.)

The return value of hcubature is 0 on success and nonzero if
there was an error (currently, only out-of-memory situations or when
the integrand signals an error).  For a nonzero return value, the
contents of the VAL and ERR arrays are undefined.

The integrand function F should be a function of the form:

int F(unsigned ndim, const double *x, void *fdata,
      unsigned fdim, double *fval);

Here, the input is an array X of length NDIM (the point to be
evaluated), the output is an array FVAL of length FDIM (the vector of
function values at the point X).  The return value should be 0 on
success or a nonzero value if an error occurred and the integration
is to be terminated immediately (hcubature will then return
a nonzero error code).

The FDATA argument of F is equal to the FDATA argument passed to
cubature -- this can be used by the caller to pass any
additional information through to F as needed (rather than using
global variables, which are not re-entrant).  If F does not need any
additional data, you can just pass FDATA = NULL and ignore the FDATA
argument to F.

-----------------------------------------------------------------------
== "Vectorized" interface ==

For parallelization and other purposes, it is useful to call a single
integrand function with an array of points to evaluate, rather than a
single point at a time.  (e.g. you may wish to evaluate the different
points in parallel.)  This is accomplished by calling:

int hcubature_v(unsigned fdim, integrand_v f, void *fdata,
                unsigned dim, const double *xmin, const double *xmax,
                unsigned maxEval, double reqAbsError, double reqRelError,
                error_norm norm, double *val, double *err);

(and similarly for pcubature_v) where all the arguments are the
same as above except that the integrand function F should now be a
function of the form:

int F(unsigned ndim, size_t npts, const double *x, void *fdata,
      unsigned fdim, double *fval);

Now, x is not a single point, but an array of npts points, and upon
return the values of all fdim integrands at all npts points should be
stored in fval.  In particular, x[i*ndim + j] is the j-th coordinate
of the i-th point (i < npts and j < ndim), and the k-th function
evaluation (k < fdim) for the i-th point is returned in fval[i*fdim + k].

Again, the return value should be 0 on success or nonzero to terminate
the integration immediately (e.g. if an error occurred).

The size of npts will vary with the dimensionality of the problem;
higher-dimensional problems will have (exponentially) larger npts,
allowing for the possibility of more parallelism.  Also npts will vary
between calls to your integrand---subsequent calls (if the first batch
of points did not achieve the desired accuracy) will have larger and
larger values of npts.

-----------------------------------------------------------------------
Test cases:

To compile a test case, just compile cubature.c while #defining
TEST_INTEGRATOR, e.g. (on Unix or GNU/Linux) via:

 cc -DHCUBATURE -o htest hcubature.c test.c -lm
 cc -DPCUBATURE -o ptest pcubature.c test.c -lm

The usage is then:

    ./htest <dim> <tol> <integrand> <maxeval>
    ./ptest <dim> <tol> <integrand> <maxeval>

where <dim> = # dimensions, <tol> = relative tolerance, <integrand> is
0-7 for one of eight possible test integrands (see below) and
<maxeval> is the maximum # function evaluations (0 for none, the default).

The different test integrands are:

0: a product of cosine functions
1: a Gaussian integral of exp(-x^2), remapped to [0,infinity) limits
2: volume of a hypersphere (integrating a discontinuous function!)
3: a simple polynomial (product of coordinates)
4: a Gaussian centered in the middle of the integration volume
5: a sum of two Gaussians
6: an example function by Tsuda, a product of terms with near poles
7: a test integrand by Morokoff and Caflisch, a simple product of
   dim-th roots of the coordinates (weakly singular at the boundary)

For example:

    ./htest 3 1e-5 4

integrates the Gaussian function (4) to a desired relative error
tolerance of 1e-5 in 3 dimensions.  The output is:

3-dim integral, tolerance = 1e-05
integrand 4: integral = 1, est err = 9.99952e-06, true err = 2.54397e-08
#evals = 82203

Notice that it finds the integral after 82203 function evaluations
with an estimated error of about 1e-5, but the true error (compared to
the exact result) is much smaller (2.5e-8): the error estimation is
typically conservative when applied to smooth functions like this.
