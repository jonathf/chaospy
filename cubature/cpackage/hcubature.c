/* Adaptive multidimensional integration of a vector of integrands.
 *
 * Copyright (c) 2005-2013 Steven G. Johnson
 *
 * Portions (see comments) based on HIntLib (also distributed under
 * the GNU GPL, v2 or later), copyright (c) 2002-2005 Rudolf Schuerer.
 *     (http://www.cosy.sbg.ac.at/~rschuer/hintlib/)
 *
 * Portions (see comments) based on GNU GSL (also distributed under
 * the GNU GPL, v2 or later), copyright (c) 1996-2000 Brian Gough.
 *     (http://www.gnu.org/software/gsl/)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>

/* Adaptive multidimensional integration on hypercubes (or, really,
   hyper-rectangles) using cubature rules.

   A cubature rule takes a function and a hypercube and evaluates
   the function at a small number of points, returning an estimate
   of the integral as well as an estimate of the error, and also
   a suggested dimension of the hypercube to subdivide.

   Given such a rule, the adaptive integration is simple:

   1) Evaluate the cubature rule on the hypercube(s).
      Stop if converged.

   2) Pick the hypercube with the largest estimated error,
      and divide it in two along the suggested dimension.

   3) Goto (1).

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
 Library as listed in the copyright notice above, on the other hand.

 I am also grateful to Dmitry Turbiner <dturbiner@alum.mit.edu>, who
 implemented an initial prototype of the "vectorized" functionality
 for evaluating multiple points in a single call (as opposed to
 multiple functions in a single call).  (Although Dmitry implemented
 a working version, I ended up re-implementing this feature from
 scratch as part of a larger code-cleanup, and in order to have
 a single code path for the vectorized and non-vectorized APIs.  I
 subsequently implemented the algorithm by Gladwell to extract
 even more parallelism by evalutating many hypercubes at once.)

 TODO:

   * Putting these routines into the GNU GSL library would be nice.

   * A Python interface would be nice.  (Also a Matlab interface,
     a GNU Octave interface, ...)

   * For high-dimensional integrals, it would be nice to implement
     a sparse-grid cubature scheme using Clenshaw-Curtis quadrature.
     Currently, for dimensions > 7 or so, quasi Monte Carlo methods win.

   * Berntsen et. al also describe a "two-level" error estimation scheme
     that they claim makes the algorithm more robust.  It might be
     nice to implement this, at least as an option (although I seem
     to remember trying it once and it made the number of evaluations
     substantially worse for my test integrands).

*/

/* USAGE: Call cubature with your function as described in cubature.h.

	  To compile a test program, compile cubature.c with
	  -DTEST_INTEGRATOR as described at the end. */

#include "cubature.h"

/* error return codes */
#define SUCCESS 0
#define FAILURE 1

/***************************************************************************/
/* Basic datatypes */

typedef struct {
     double val, err;
} esterr;

static double errMax(unsigned fdim, const esterr *ee)
{
     double errmax = 0;
     unsigned k;
     for (k = 0; k < fdim; ++k)
	  if (ee[k].err > errmax) errmax = ee[k].err;
     return errmax;
}

typedef struct {
     unsigned dim;
     double *data;	/* length 2*dim = center followed by half-widths */
     double vol;	/* cache volume = product of widths */
} hypercube;

static double compute_vol(const hypercube *h)
{
     unsigned i;
     double vol = 1;
     for (i = 0; i < h->dim; ++i)
	  vol *= 2 * h->data[i + h->dim];
     return vol;
}

static hypercube make_hypercube(unsigned dim, const double *center, const double *halfwidth)
{
     unsigned i;
     hypercube h;
     h.dim = dim;
     h.data = (double *) malloc(sizeof(double) * dim * 2);
     h.vol = 0;
     if (h.data) {
	  for (i = 0; i < dim; ++i) {
	       h.data[i] = center[i];
	       h.data[i + dim] = halfwidth[i];
	  }
	  h.vol = compute_vol(&h);
     }
     return h;
}

static hypercube make_hypercube_range(unsigned dim, const double *xmin, const double *xmax)
{
     hypercube h = make_hypercube(dim, xmin, xmax);
     unsigned i;
     if (h.data) {
	  for (i = 0; i < dim; ++i) {
	       h.data[i] = 0.5 * (xmin[i] + xmax[i]);
	       h.data[i + dim] = 0.5 * (xmax[i] - xmin[i]);
	  }
	  h.vol = compute_vol(&h);
     }
     return h;
}

static void destroy_hypercube(hypercube *h)
{
     free(h->data);
     h->dim = 0;
}

typedef struct {
     hypercube h;
     unsigned splitDim;
     unsigned fdim; /* dimensionality of vector integrand */
     esterr *ee; /* array of length fdim */
     double errmax; /* max ee[k].err */
} region;

static region make_region(const hypercube *h, unsigned fdim)
{
     region R;
     R.h = make_hypercube(h->dim, h->data, h->data + h->dim);
     R.splitDim = 0;
     R.fdim = fdim;
     R.ee = R.h.data ? (esterr *) malloc(sizeof(esterr) * fdim) : NULL;
     return R;
}

static void destroy_region(region *R)
{
     destroy_hypercube(&R->h);
     free(R->ee);
     R->ee = 0;
}

static int cut_region(region *R, region *R2)
{
     unsigned d = R->splitDim, dim = R->h.dim;
     *R2 = *R;
     R->h.data[d + dim] *= 0.5;
     R->h.vol *= 0.5;
     R2->h = make_hypercube(dim, R->h.data, R->h.data + dim);
     if (!R2->h.data) return FAILURE;
     R->h.data[d] -= R->h.data[d + dim];
     R2->h.data[d] += R->h.data[d + dim];
     R2->ee = (esterr *) malloc(sizeof(esterr) * R2->fdim);
     return R2->ee == NULL;
}

struct rule_s; /* forward declaration */

typedef int (*evalError_func)(struct rule_s *r,
			      unsigned fdim, integrand_v f, void *fdata,
			      unsigned nR, region *R);
typedef void (*destroy_func)(struct rule_s *r);


typedef struct rule_s {
     unsigned dim, fdim;         /* the dimensionality & number of functions */
     unsigned num_points;       /* number of evaluation points */
     unsigned num_regions; /* max number of regions evaluated at once */
     double *pts; /* points to eval: num_regions * num_points * dim */
     double *vals; /* num_regions * num_points * fdim */
     evalError_func evalError;
     destroy_func destroy;
} rule;

static void destroy_rule(rule *r)
{
     if (r) {
	  if (r->destroy) r->destroy(r);
	  free(r->pts);
	  free(r);
     }
}

static int alloc_rule_pts(rule *r, unsigned num_regions)
{
     if (num_regions > r->num_regions) {
	  free(r->pts);
	  r->pts = r->vals = NULL;
	  r->num_regions = 0;
	  num_regions *= 2; /* allocate extra so that
			       repeatedly calling alloc_rule_pts with
			       growing num_regions only needs
			       a logarithmic number of allocations */
	  r->pts = (double *) malloc(sizeof(double) * 
				     (num_regions
				      * r->num_points * (r->dim + r->fdim)));
	  if (r->fdim + r->dim > 0 && !r->pts) return FAILURE;
	  r->vals = r->pts + num_regions * r->num_points * r->dim;
	  r->num_regions = num_regions;
     }
     return SUCCESS;
}

static rule *make_rule(size_t sz, /* >= sizeof(rule) */
		       unsigned dim, unsigned fdim, unsigned num_points,
		       evalError_func evalError, destroy_func destroy)
{
     rule *r;

     if (sz < sizeof(rule)) return NULL;
     r = (rule *) malloc(sz);
     if (!r) return NULL;
     r->pts = r->vals = NULL;
     r->num_regions = 0;
     r->dim = dim; r->fdim = fdim; r->num_points = num_points;
     r->evalError = evalError;
     r->destroy = destroy;
     return r;
}

/* note: all regions must have same fdim */
static int eval_regions(unsigned nR, region *R, 
			integrand_v f, void *fdata, rule *r)
{
     unsigned iR;
     if (nR == 0) return SUCCESS; /* nothing to evaluate */
     if (r->evalError(r, R->fdim, f, fdata, nR, R)) return FAILURE;
     for (iR = 0; iR < nR; ++iR)
	  R[iR].errmax = errMax(R->fdim, R[iR].ee);
     return SUCCESS;
}

/***************************************************************************/
/* Functions to loop over points in a hypercube. */

/* Based on orbitrule.cpp in HIntLib-0.0.10 */

/* ls0 returns the least-significant 0 bit of n (e.g. it returns
   0 if the LSB is 0, it returns 1 if the 2 LSBs are 01, etcetera). */

static unsigned ls0(unsigned n)
{
#if defined(__GNUC__) && \
    ((__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || __GNUC__ > 3)
     return __builtin_ctz(~n); /* gcc builtin for version >= 3.4 */
#else
     const unsigned bits[256] = {
	  0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
	  0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
	  0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
	  0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6,
	  0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
	  0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
	  0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
	  0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7,
	  0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
	  0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
	  0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
	  0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6,
	  0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
	  0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
	  0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
	  0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 8,
     };
     unsigned bit = 0;
     while ((n & 0xff) == 0xff) {
	  n >>= 8;
	  bit += 8;
     }
     return bit + bits[n & 0xff];
#endif
}

/**
 *  Evaluate the integration points for all 2^n points (+/-r,...+/-r)
 *
 *  A Gray-code ordering is used to minimize the number of coordinate updates
 *  in p, although this doesn't matter as much now that we are saving all pts.
 */
static void evalR_Rfs(double *pts, unsigned dim, double *p, const double *c, const double *r)
{
     unsigned i;
     unsigned signs = 0; /* 0/1 bit = +/- for corresponding element of r[] */

     /* We start with the point where r is ADDed in every coordinate
        (this implies signs=0). */
     for (i = 0; i < dim; ++i)
	  p[i] = c[i] + r[i];

     /* Loop through the points in Gray-code ordering */
     for (i = 0;; ++i) {
	  unsigned mask, d;

	  memcpy(pts, p, sizeof(double) * dim); pts += dim;

	  d = ls0(i);	/* which coordinate to flip */
	  if (d >= dim)
	       break;

	  /* flip the d-th bit and add/subtract r[d] */
	  mask = 1U << d;
	  signs ^= mask;
	  p[d] = (signs & mask) ? c[d] - r[d] : c[d] + r[d];
     }
}

static void evalRR0_0fs(double *pts, unsigned dim, double *p, const double *c, const double *r)
{
     unsigned i, j;

     for (i = 0; i < dim - 1; ++i) {
	  p[i] = c[i] - r[i];
	  for (j = i + 1; j < dim; ++j) {
	       p[j] = c[j] - r[j];
	       memcpy(pts, p, sizeof(double) * dim); pts += dim;
	       p[i] = c[i] + r[i];
	       memcpy(pts, p, sizeof(double) * dim); pts += dim;
	       p[j] = c[j] + r[j];
	       memcpy(pts, p, sizeof(double) * dim); pts += dim;
	       p[i] = c[i] - r[i];
	       memcpy(pts, p, sizeof(double) * dim); pts += dim;

	       p[j] = c[j];	/* Done with j -> Restore p[j] */
	  }
	  p[i] = c[i];		/* Done with i -> Restore p[i] */
     }
}

static void evalR0_0fs4d(double *pts, unsigned dim, double *p, const double *c,
			 const double *r1, const double *r2)
{
     unsigned i;

     memcpy(pts, p, sizeof(double) * dim); pts += dim;

     for (i = 0; i < dim; i++) {
	  p[i] = c[i] - r1[i];
	  memcpy(pts, p, sizeof(double) * dim); pts += dim;

	  p[i] = c[i] + r1[i];
	  memcpy(pts, p, sizeof(double) * dim); pts += dim;

	  p[i] = c[i] - r2[i];
	  memcpy(pts, p, sizeof(double) * dim); pts += dim;

	  p[i] = c[i] + r2[i];
	  memcpy(pts, p, sizeof(double) * dim); pts += dim;

	  p[i] = c[i];
     }
}

#define num0_0(dim) (1U)
#define numR0_0fs(dim) (2 * (dim))
#define numRR0_0fs(dim) (2 * (dim) * (dim-1))
#define numR_Rfs(dim) (1U << (dim))

/***************************************************************************/
/* Based on rule75genzmalik.cpp in HIntLib-0.0.10: An embedded
   cubature rule of degree 7 (embedded rule degree 5) due to A. C. Genz
   and A. A. Malik.  See:

         A. C. Genz and A. A. Malik, "An imbedded [sic] family of fully
         symmetric numerical integration rules," SIAM
         J. Numer. Anal. 20 (3), 580-588 (1983).
*/

typedef struct {
     rule parent;

     /* temporary arrays of length dim */
     double *widthLambda, *widthLambda2, *p;

     /* dimension-dependent constants */
     double weight1, weight3, weight5;
     double weightE1, weightE3;
} rule75genzmalik;

#define real(x) ((double)(x))
#define to_int(n) ((int)(n))

static int isqr(int x)
{
     return x * x;
}

static void destroy_rule75genzmalik(rule *r_)
{
     rule75genzmalik *r = (rule75genzmalik *) r_;
     free(r->p);
}

static int rule75genzmalik_evalError(rule *r_, unsigned fdim, integrand_v f, void *fdata, unsigned nR, region *R)
{
     /* lambda2 = sqrt(9/70), lambda4 = sqrt(9/10), lambda5 = sqrt(9/19) */
     const double lambda2 = 0.3585685828003180919906451539079374954541;
     const double lambda4 = 0.9486832980505137995996680633298155601160;
     const double lambda5 = 0.6882472016116852977216287342936235251269;
     const double weight2 = 980. / 6561.;
     const double weight4 = 200. / 19683.;
     const double weightE2 = 245. / 486.;
     const double weightE4 = 25. / 729.;
     const double ratio = (lambda2 * lambda2) / (lambda4 * lambda4);

     rule75genzmalik *r = (rule75genzmalik *) r_;
     unsigned i, j, iR, dim = r_->dim;
     size_t npts = 0;
     double *diff, *pts, *vals;

     if (alloc_rule_pts(r_, nR)) return FAILURE;
     pts = r_->pts; vals = r_->vals;

     for (iR = 0; iR < nR; ++iR) {
	  const double *center = R[iR].h.data;
	  const double *halfwidth = R[iR].h.data + dim;
	  
	  for (i = 0; i < dim; ++i)
	       r->p[i] = center[i];
	  
	  for (i = 0; i < dim; ++i)
	       r->widthLambda2[i] = halfwidth[i] * lambda2;
	  for (i = 0; i < dim; ++i)
	       r->widthLambda[i] = halfwidth[i] * lambda4;

	  /* Evaluate points in the center, in (lambda2,0,...,0) and
	     (lambda3=lambda4, 0,...,0).  */
	  evalR0_0fs4d(pts + npts*dim, dim, r->p, center, 
		       r->widthLambda2, r->widthLambda);
	  npts += num0_0(dim) + 2 * numR0_0fs(dim);

	  /* Calculate points for (lambda4, lambda4, 0, ...,0) */
	  evalRR0_0fs(pts + npts*dim, dim, r->p, center, r->widthLambda);
	  npts += numRR0_0fs(dim);

	  /* Calculate points for (lambda5, lambda5, ..., lambda5) */
	  for (i = 0; i < dim; ++i)
	       r->widthLambda[i] = halfwidth[i] * lambda5;
	  evalR_Rfs(pts + npts*dim, dim, r->p, center, r->widthLambda);
	  npts += numR_Rfs(dim);
     }

     /* Evaluate the integrand function(s) at all the points */
     if (f(dim, npts, pts, fdata, fdim, vals))
	  return FAILURE;

     /* we are done with the points, and so we can re-use the pts
	array to store the maximum difference diff[i] in each dimension 
	for each hypercube */
     diff = pts;
     for (i = 0; i < dim * nR; ++i) diff[i] = 0;

     for (j = 0; j < fdim; ++j) {
	  const double *v = vals + j;
#         define VALS(i) v[fdim*(i)]	       
	  for (iR = 0; iR < nR; ++iR) {
	       double result, res5th;
	       double val0, sum2=0, sum3=0, sum4=0, sum5=0;
	       unsigned k, k0 = 0;
	       /* accumulate j-th function values into j-th integrals
		     NOTE: this relies on the ordering of the eval functions
		     above, as well as on the internal structure of
		     the evalR0_0fs4d function */

	       val0 = VALS(0); /* central point */
	       k0 += 1;

	       for (k = 0; k < dim; ++k) {
		    double v0 = VALS(k0 + 4*k);
		    double v1 = VALS((k0 + 4*k) + 1);
		    double v2 = VALS((k0 + 4*k) + 2);
		    double v3 = VALS((k0 + 4*k) + 3);
		    
		    sum2 += v0 + v1;
		    sum3 += v2 + v3;
		    
		    diff[iR * dim + k] += 
			 fabs(v0 + v1 - 2*val0 - ratio * (v2 + v3 - 2*val0));
	       }
	       k0 += 4*k;

	       for (k = 0; k < numRR0_0fs(dim); ++k)
		    sum4 += VALS(k0 + k);
	       k0 += k;
	       
	       for (k = 0; k < numR_Rfs(dim); ++k)
		    sum5 += VALS(k0 + k);
	       
	       /* Calculate fifth and seventh order results */
	       result = R[iR].h.vol * (r->weight1 * val0 + weight2 * sum2 + r->weight3 * sum3 + weight4 * sum4 + r->weight5 * sum5);
	       res5th = R[iR].h.vol * (r->weightE1 * val0 + weightE2 * sum2 + r->weightE3 * sum3 + weightE4 * sum4);
	       
	       R[iR].ee[j].val = result;
	       R[iR].ee[j].err = fabs(res5th - result);
	       
	       v += r_->num_points * fdim;
	  }
#         undef VALS
     }


     /* figure out dimension to split: */
     for (iR = 0; iR < nR; ++iR) {
	  double maxdiff = 0;
	  unsigned dimDiffMax = 0;

	  for (i = 0; i < dim; ++i)
	       if (diff[iR*dim + i] > maxdiff) {
		    maxdiff = diff[iR*dim + i];
		    dimDiffMax = i;
	       }
	  R[iR].splitDim = dimDiffMax;
     }
     return SUCCESS;
}

static rule *make_rule75genzmalik(unsigned dim, unsigned fdim)
{
     rule75genzmalik *r;

     if (dim < 2) return NULL; /* this rule does not support 1d integrals */

     /* Because of the use of a bit-field in evalR_Rfs, we are limited
	to be < 32 dimensions (or however many bits are in unsigned).
	This is not a practical limitation...long before you reach
	32 dimensions, the Genz-Malik cubature becomes excruciatingly
	slow and is superseded by other methods (e.g. Monte-Carlo). */
     if (dim >= sizeof(unsigned) * 8) return NULL;

     r = (rule75genzmalik *) make_rule(sizeof(rule75genzmalik),
				       dim, fdim,
				       num0_0(dim) + 2 * numR0_0fs(dim)
				       + numRR0_0fs(dim) + numR_Rfs(dim),
				       rule75genzmalik_evalError,
				       destroy_rule75genzmalik);
     if (!r) return NULL;

     r->weight1 = (real(12824 - 9120 * to_int(dim) + 400 * isqr(to_int(dim)))
		   / real(19683));
     r->weight3 = real(1820 - 400 * to_int(dim)) / real(19683);
     r->weight5 = real(6859) / real(19683) / real(1U << dim);
     r->weightE1 = (real(729 - 950 * to_int(dim) + 50 * isqr(to_int(dim)))
		    / real(729));
     r->weightE3 = real(265 - 100 * to_int(dim)) / real(1458);

     r->p = (double *) malloc(sizeof(double) * dim * 3);
     if (!r->p) { destroy_rule((rule *) r); return NULL; }
     r->widthLambda = r->p + dim;
     r->widthLambda2 = r->p + 2 * dim;

     return (rule *) r;
}

/***************************************************************************/
/* 1d 15-point Gaussian quadrature rule, based on qk15.c and qk.c in
   GNU GSL (which in turn is based on QUADPACK). */

static int rule15gauss_evalError(rule *r,
				 unsigned fdim, integrand_v f, void *fdata,
				 unsigned nR, region *R)
{
     /* Gauss quadrature weights and kronrod quadrature abscissae and
	weights as evaluated with 80 decimal digit arithmetic by
	L. W. Fullerton, Bell Labs, Nov. 1981. */
     const unsigned n = 8;
     const double xgk[8] = {  /* abscissae of the 15-point kronrod rule */
	  0.991455371120812639206854697526329,
	  0.949107912342758524526189684047851,
	  0.864864423359769072789712788640926,
	  0.741531185599394439863864773280788,
	  0.586087235467691130294144838258730,
	  0.405845151377397166906606412076961,
	  0.207784955007898467600689403773245,
	  0.000000000000000000000000000000000
	  /* xgk[1], xgk[3], ... abscissae of the 7-point gauss rule. 
	     xgk[0], xgk[2], ... to optimally extend the 7-point gauss rule */
     };
     static const double wg[4] = {  /* weights of the 7-point gauss rule */
	  0.129484966168869693270611432679082,
	  0.279705391489276667901467771423780,
	  0.381830050505118944950369775488975,
	  0.417959183673469387755102040816327
     };
     static const double wgk[8] = { /* weights of the 15-point kronrod rule */
	  0.022935322010529224963732008058970,
	  0.063092092629978553290700663189204,
	  0.104790010322250183839876322541518,
	  0.140653259715525918745189590510238,
	  0.169004726639267902826583426598550,
	  0.190350578064785409913256402421014,
	  0.204432940075298892414161999234649,
	  0.209482141084727828012999174891714
     };
     unsigned j, k, iR;
     size_t npts = 0;
     double *pts, *vals;

     if (alloc_rule_pts(r, nR)) return FAILURE;
     pts = r->pts; vals = r->vals;

     for (iR = 0; iR < nR; ++iR) {
	  const double center = R[iR].h.data[0];
	  const double halfwidth = R[iR].h.data[1];

	  pts[npts++] = center;

	  for (j = 0; j < (n - 1) / 2; ++j) {
	       int j2 = 2*j + 1;
	       double w = halfwidth * xgk[j2];
	       pts[npts++] = center - w;
	       pts[npts++] = center + w;
	  }
	  for (j = 0; j < n/2; ++j) {
	       int j2 = 2*j;
	       double w = halfwidth * xgk[j2];
	       pts[npts++] = center - w;
	       pts[npts++] = center + w;
	  }

	  R[iR].splitDim = 0; /* no choice but to divide 0th dimension */
     }

     if (f(1, npts, pts, fdata, fdim, vals))
	  return FAILURE;
     
     for (k = 0; k < fdim; ++k) {
          const double *vk = vals + k;
	  for (iR = 0; iR < nR; ++iR) {
	       const double halfwidth = R[iR].h.data[1];
	       double result_gauss = vk[0] * wg[n/2 - 1];
	       double result_kronrod = vk[0] * wgk[n - 1];
	       double result_abs = fabs(result_kronrod);
	       double result_asc, mean, err;

	       /* accumulate integrals */
	       npts = 1;
	       for (j = 0; j < (n - 1) / 2; ++j) {
		    int j2 = 2*j + 1;
		    double v = vk[fdim*npts] + vk[fdim*npts+fdim];
		    result_gauss += wg[j] * v;
		    result_kronrod += wgk[j2] * v;
		    result_abs += wgk[j2] * (fabs(vk[fdim*npts]) 
					     + fabs(vk[fdim*npts+fdim]));
		    npts += 2;
	       }
	       for (j = 0; j < n/2; ++j) {
		    int j2 = 2*j;
		    result_kronrod += wgk[j2] * (vk[fdim*npts] 
						 + vk[fdim*npts+fdim]);
		    result_abs += wgk[j2] * (fabs(vk[fdim*npts]) 
					     + fabs(vk[fdim*npts+fdim]));
		    npts += 2;
	       }
	       
	       /* integration result */
	       R[iR].ee[k].val = result_kronrod * halfwidth;

	       /* error estimate 
		  (from GSL, probably dates back to QUADPACK
		  ... not completely clear to me why we don't just use
	          fabs(result_kronrod - result_gauss) * halfwidth */
	       mean = result_kronrod * 0.5;
	       result_asc = wgk[n - 1] * fabs(vk[0] - mean);
	       npts = 1;
	       for (j = 0; j < (n - 1) / 2; ++j) {
		    int j2 = 2*j + 1;
		    result_asc += wgk[j2] * (fabs(vk[fdim*npts]-mean)
					     + fabs(vk[fdim*npts+fdim]-mean));
		    npts += 2;
	       }
	       for (j = 0; j < n/2; ++j) {
		    int j2 = 2*j;
		    result_asc += wgk[j2] * (fabs(vk[fdim*npts]-mean)
					     + fabs(vk[fdim*npts+fdim]-mean));
		    npts += 2;
	       }
	       err = fabs(result_kronrod - result_gauss) * halfwidth;
	       result_abs *= halfwidth;
	       result_asc *= halfwidth;
	       if (result_asc != 0 && err != 0) {
		    double scale = pow((200 * err / result_asc), 1.5);
		    err = (scale < 1) ? result_asc * scale : result_asc;
	       }
	       if (result_abs > DBL_MIN / (50 * DBL_EPSILON)) {
		    double min_err = 50 * DBL_EPSILON * result_abs;
		    if (min_err > err) err = min_err;
	       }
	       R[iR].ee[k].err = err;
	       
	       /* increment vk to point to next batch of results */
	       vk += 15*fdim;
	  }
     }
     return SUCCESS;
}

static rule *make_rule15gauss(unsigned dim, unsigned fdim)
{
     if (dim != 1) return NULL; /* this rule is only for 1d integrals */

     return make_rule(sizeof(rule), dim, fdim, 15,
		      rule15gauss_evalError, 0);
}

/***************************************************************************/
/* binary heap implementation (ala _Introduction to Algorithms_ by
   Cormen, Leiserson, and Rivest), for use as a priority queue of
   regions to integrate. */

typedef region heap_item;
#define KEY(hi) ((hi).errmax)

typedef struct {
     size_t n, nalloc;
     heap_item *items;
     unsigned fdim;
     esterr *ee; /* array of length fdim of the total integrand & error */
} heap;

static void heap_resize(heap *h, size_t nalloc)
{
     h->nalloc = nalloc;
     h->items = (heap_item *) realloc(h->items, sizeof(heap_item) * nalloc);
}

static heap heap_alloc(size_t nalloc, unsigned fdim)
{
     heap h;
     unsigned i;
     h.n = 0;
     h.nalloc = 0;
     h.items = 0;
     h.fdim = fdim;
     h.ee = (esterr *) malloc(sizeof(esterr) * fdim);
     if (h.ee) {
	  for (i = 0; i < fdim; ++i) h.ee[i].val = h.ee[i].err = 0;
	  heap_resize(&h, nalloc);
     }
     return h;
}

/* note that heap_free does not deallocate anything referenced by the items */
static void heap_free(heap *h)
{
     h->n = 0;
     heap_resize(h, 0);
     h->fdim = 0;
     free(h->ee);
}

static int heap_push(heap *h, heap_item hi)
{
     int insert;
     unsigned i, fdim = h->fdim;

     for (i = 0; i < fdim; ++i) {
	  h->ee[i].val += hi.ee[i].val;
	  h->ee[i].err += hi.ee[i].err;
     }
     insert = h->n;
     if (++(h->n) > h->nalloc) {
	  heap_resize(h, h->n * 2);
	  if (!h->items) return FAILURE;
     }

     while (insert) {
	  int parent = (insert - 1) / 2;
	  if (KEY(hi) <= KEY(h->items[parent]))
	       break;
	  h->items[insert] = h->items[parent];
	  insert = parent;
     }
     h->items[insert] = hi;
     return SUCCESS;
}

static int heap_push_many(heap *h, size_t ni, heap_item *hi)
{
     size_t i;
     for (i = 0; i < ni; ++i)
	  if (heap_push(h, hi[i])) return FAILURE;
     return SUCCESS;
}

static heap_item heap_pop(heap *h)
{
     heap_item ret;
     int i, n, child;

     if (!(h->n)) {
	  fprintf(stderr, "attempted to pop an empty heap\n");
	  exit(EXIT_FAILURE);
     }

     ret = h->items[0];
     h->items[i = 0] = h->items[n = --(h->n)];
     while ((child = i * 2 + 1) < n) {
	  int largest;
	  heap_item swap;

	  if (KEY(h->items[child]) <= KEY(h->items[i]))
	       largest = i;
	  else
	       largest = child;
	  if (++child < n && KEY(h->items[largest]) < KEY(h->items[child]))
	       largest = child;
	  if (largest == i)
	       break;
	  swap = h->items[i];
	  h->items[i] = h->items[largest];
	  h->items[i = largest] = swap;
     }

     {
	  unsigned i, fdim = h->fdim;
	  for (i = 0; i < fdim; ++i) {
	       h->ee[i].val -= ret.ee[i].val;
	       h->ee[i].err -= ret.ee[i].err;
	  }
     }
     return ret;
}

/***************************************************************************/

static int hconverged(unsigned fdim, const esterr *ee,
		     double reqAbsError, double reqRelError, error_norm norm)
#define ERR(j) ee[j].err
#define VAL(j) ee[j].val
#include "converged.h"

/***************************************************************************/

/* adaptive integration, analogous to adaptintegrator.cpp in HIntLib */

static int rulecubature(rule *r, unsigned fdim, 
			integrand_v f, void *fdata, 
			const hypercube *h, 
			size_t maxEval,
			double reqAbsError, double reqRelError,
			error_norm norm,
			double *val, double *err, int parallel)
{
     size_t numEval = 0;
     heap regions;
     unsigned i, j;
     region *R = NULL; /* array of regions to evaluate */
     size_t nR_alloc = 0;
     esterr *ee = NULL;

     if (fdim <= 1) norm = ERROR_INDIVIDUAL; /* norm is irrelevant */
     if (norm < 0 || norm > ERROR_LINF) return FAILURE; /* invalid norm */

     regions = heap_alloc(1, fdim);
     if (!regions.ee || !regions.items) goto bad;

     ee = (esterr *) malloc(sizeof(esterr) * fdim);
     if (!ee) goto bad;
     
     nR_alloc = 2;
     R = (region *) malloc(sizeof(region) * nR_alloc);
     if (!R) goto bad;
     R[0] = make_region(h, fdim);
     if (!R[0].ee
	 || eval_regions(1, R, f, fdata, r)
	 || heap_push(&regions, R[0]))
	       goto bad;
     numEval += r->num_points;
     
     while (numEval < maxEval || !maxEval) {
	  if (hconverged(fdim, regions.ee, reqAbsError, reqRelError, norm))
	       break;

	  if (parallel) { /* maximize potential parallelism */
	       /* adapted from I. Gladwell, "Vectorization of one
		  dimensional quadrature codes," pp. 230--238 in
		  _Numerical Integration. Recent Developments,
		  Software and Applications_, G. Fairweather and
		  P. M. Keast, eds., NATO ASI Series C203, Dordrecht
		  (1987), as described in J. M. Bull and
		  T. L. Freeman, "Parallel Globally Adaptive
		  Algorithms for Multi-dimensional Integration,"
		  http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.6638
		  (1994). 

		  Basically, this evaluates in one shot all regions
		  that *must* be evaluated in order to reduce the
		  error to the requested bound: the minimum set of
		  largest-error regions whose errors push the total
		  error over the bound.

		  [Note: Bull and Freeman claim that the Gladwell
		  approach is intrinsically inefficent because it
		  "requires sorting", and propose an alternative
		  algorithm that "only" requires three passes over the
		  entire set of regions.  Apparently, they didn't
		  realize that one could use a heap data structure, in
		  which case the time to pop K biggest-error regions
		  out of N is only O(K log N), much better than the
		  O(N) cost of the Bull and Freeman algorithm if K <<
		  N, and it is also much simpler.] */
	       size_t nR = 0;
	       for (j = 0; j < fdim; ++j) ee[j] = regions.ee[j];
	       do {
		    if (nR + 2 > nR_alloc) {
			 nR_alloc = (nR + 2) * 2;
			 R = (region *) realloc(R, nR_alloc * sizeof(region));
			 if (!R) goto bad;
		    }
		    R[nR] = heap_pop(&regions);
		    for (j = 0; j < fdim; ++j) ee[j].err -= R[nR].ee[j].err;
		    if (cut_region(R+nR, R+nR+1)) goto bad;
		    numEval += r->num_points * 2;
		    nR += 2;
		    if (hconverged(fdim, ee, reqAbsError, reqRelError, norm))
			 break; /* other regions have small errs */
	       } while (regions.n > 0 && (numEval < maxEval || !maxEval));
	       if (eval_regions(nR, R, f, fdata, r)
		   || heap_push_many(&regions, nR, R))
		    goto bad;
	  }
	  else { /* minimize number of function evaluations */
	       R[0] = heap_pop(&regions); /* get worst region */
	       if (cut_region(R, R+1)
		   || eval_regions(2, R, f, fdata, r)
		   || heap_push_many(&regions, 2, R))
		    goto bad;
	       numEval += r->num_points * 2;
	  }
     }

     /* re-sum integral and errors */
     for (j = 0; j < fdim; ++j) val[j] = err[j] = 0;  
     for (i = 0; i < regions.n; ++i) {
	  for (j = 0; j < fdim; ++j) { 
	       val[j] += regions.items[i].ee[j].val;
	       err[j] += regions.items[i].ee[j].err;
	  }
	  destroy_region(&regions.items[i]);
     }

     /* printf("regions.nalloc = %d\n", regions.nalloc); */
     free(ee);
     heap_free(&regions);
     free(R);
     return SUCCESS;

bad:
     free(ee);
     heap_free(&regions);
     free(R);
     return FAILURE;
}

static int cubature(unsigned fdim, integrand_v f, void *fdata, 
		    unsigned dim, const double *xmin, const double *xmax, 
		    size_t maxEval, double reqAbsError, double reqRelError, 
		    error_norm norm,
		    double *val, double *err, int parallel)
{
     rule *r;
     hypercube h;
     int status;
     unsigned i;
     
     if (fdim == 0) /* nothing to do */ return SUCCESS;
     if (dim == 0) { /* trivial integration */
	  if (f(0, 1, xmin, fdata, fdim, val)) return FAILURE;
	  for (i = 0; i < fdim; ++i) err[i] = 0;
	  return SUCCESS;
     }
     r = dim == 1 ? make_rule15gauss(dim, fdim)
 	          : make_rule75genzmalik(dim, fdim);
     if (!r) { 
	  for (i = 0; i < fdim; ++i) {
	       val[i] = 0;
	       err[i] = HUGE_VAL; 
	  }
	  return FAILURE;
     }
     h = make_hypercube_range(dim, xmin, xmax);
     status = !h.data ? FAILURE
	  : rulecubature(r, fdim, f, fdata, &h,
				maxEval, reqAbsError, reqRelError, norm,
				val, err, parallel);
     destroy_hypercube(&h);
     destroy_rule(r);
     return status;
}

int hcubature_v(unsigned fdim, integrand_v f, void *fdata, 
                unsigned dim, const double *xmin, const double *xmax, 
                size_t maxEval, double reqAbsError, double reqRelError, 
                error_norm norm,
                double *val, double *err)
{
     return cubature(fdim, f, fdata, dim, xmin, xmax, 
		     maxEval, reqAbsError, reqRelError, norm, val, err, 1);
}

#include "vwrapper.h"

int hcubature(unsigned fdim, integrand f, void *fdata, 
	      unsigned dim, const double *xmin, const double *xmax, 
	      size_t maxEval, double reqAbsError, double reqRelError, 
	      error_norm norm,
	      double *val, double *err)
{
     int ret;
     fv_data d;

     if (fdim == 0) return SUCCESS; /* nothing to do */     
     
     d.f = f; d.fdata = fdata;
     ret = cubature(fdim, fv, &d, dim, xmin, xmax, 
		    maxEval, reqAbsError, reqRelError, norm, val, err, 0);
     return ret;
}

/***************************************************************************/
