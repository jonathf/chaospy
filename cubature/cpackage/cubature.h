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

#ifndef CUBATURE_H
#define CUBATURE_H

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

/* USAGE: Call hcubature or pcubature with your function as described
          in the README file. */

/* a vector integrand - evaluates the function at the given point x
   (an array of length ndim) and returns the result in fval (an array
   of length fdim).   The void* parameter is there in case you have
   to pass any additional data through to your function (it corresponds
   to the fdata parameter you pass to cubature).  Return 0 on
   success or nonzero to terminate the integration. */
typedef int (*integrand) (unsigned ndim, const double *x, void *,
                          unsigned fdim, double *fval);

/* a vector integrand of a vector of npt points: x[i*ndim + j] is the
   j-th coordinate of the i-th point, and the k-th function evaluation
   for the i-th point is returned in fval[i*fdim + k].  Return 0 on success
   or nonzero to terminate the integration. */
typedef int (*integrand_v) (unsigned ndim, size_t npt,
			    const double *x, void *,
			    unsigned fdim, double *fval);

/* Different ways of measuring the absolute and relative error when
   we have multiple integrands, given a vector e of error estimates
   in the individual components of a vector v of integrands.  These
   are all equivalent when there is only a single integrand. */
typedef enum {
     ERROR_INDIVIDUAL = 0, /* individual relerr criteria in each component */
     ERROR_PAIRED, /* paired L2 norms of errors in each component,
		      mainly for integrating vectors of complex numbers */
     ERROR_L2, /* abserr is L_2 norm |e|, and relerr is |e|/|v| */
     ERROR_L1, /* abserr is L_1 norm |e|, and relerr is |e|/|v| */
     ERROR_LINF /* abserr is L_\infty norm |e|, and relerr is |e|/|v| */
} error_norm;

/* Integrate the function f from xmin[dim] to xmax[dim], with at most
   maxEval function evaluations (0 for no limit), until the given
   absolute or relative error is achieved.  val returns the integral,
   and err returns the estimate for the absolute error in val; both
   of these are arrays of length fdim, the dimension of the vector
   integrand f(x). The return value of the function is 0 on success
   and non-zero if there  was an error. */

/* adapative integration by partitioning the integration domain ("h-adaptive")
   and using the same fixed-degree quadrature in each subdomain, recursively,
   until convergence is achieved. */
int hcubature(unsigned fdim, integrand f, void *fdata,
	      unsigned dim, const double *xmin, const double *xmax, 
	      size_t maxEval, double reqAbsError, double reqRelError, 
	      error_norm norm,
	      double *val, double *err);

/* as hcubature, but vectorized integrand */
int hcubature_v(unsigned fdim, integrand_v f, void *fdata,
		unsigned dim, const double *xmin, const double *xmax, 
		size_t maxEval, double reqAbsError, double reqRelError, 
		error_norm norm,
		double *val, double *err);

/* adaptive integration by increasing the degree of (tensor-product
   Clenshaw-Curtis) quadrature rules ("p-adaptive"), rather than
   subdividing the domain ("h-adaptive").  Possibly better for
   smooth integrands in low dimensions. */
int pcubature_v_buf(unsigned fdim, integrand_v f, void *fdata,
		    unsigned dim, const double *xmin, const double *xmax,
		    size_t maxEval, 
		    double reqAbsError, double reqRelError,
		    error_norm norm,
		    unsigned *m,
		    double **buf, size_t *nbuf, size_t max_nbuf,
		    double *val, double *err);
int pcubature_v(unsigned fdim, integrand_v f, void *fdata,
		unsigned dim, const double *xmin, const double *xmax, 
		size_t maxEval, double reqAbsError, double reqRelError, 
		error_norm norm,
		double *val, double *err);
int pcubature(unsigned fdim, integrand f, void *fdata,
	      unsigned dim, const double *xmin, const double *xmax, 
	      size_t maxEval, double reqAbsError, double reqRelError, 
	      error_norm norm,
	      double *val, double *err);

#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */

#endif /* CUBATURE_H */
