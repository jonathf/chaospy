/* Adaptive multidimensional integration of a vector of integrands.
 *
 * Copyright (c) 2005-2013 Steven G. Johnson
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

/* p-adaptive cubature (adaptive by increasing the degree of the
   cubature rule rather than subdividing the domain), using products
   of Clenshaw-Curtis rules.  This algorithm may be superior to
   Genz-Malik for smooth integrands lacking strongly-localized
   features, in moderate dimensions. */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cubature.h"

/* error return codes */
#define SUCCESS 0
#define FAILURE 1

/* pre-generated Clenshaw-Curtis rules and weights */
#include "clencurt.h"

/* no point in supporting very high dimensional integrals here */
#define MAXDIM (20U)

/***************************************************************************/
/* For adaptive cubature, thanks to the nesting of the C-C rules, we
   can re-use the values from coarser grids for finer grids, and the
   coarser grids are also used for error estimation. 

   A grid is determined by an m[dim] array, where m[i] denotes
   2^(m[i]+1)+1 points in the i-th dimension.
*/

/* cache of the values for the m[dim] grid.  If mi < dim, then we only
   store the values corresponding to the difference between the m grid
   and the grid with m[mi] -> m[mi]-1.  (m[mi]-1 == -1 corresponds to
   the trivial grid of one point in the center.) */
typedef struct cacheval_s {
     unsigned m[MAXDIM];
     unsigned mi;
     double *val;
} cacheval;

/* array of ncache cachevals c[i] */
typedef struct valcache_s {
     size_t ncache;
     cacheval *c;
} valcache;

static void free_cachevals(valcache *v)
{
     if (!v) return;
     if (v->c) {
	  size_t i;
	  for (i = 0; i < v->ncache; ++i)
	       free(v->c[i].val);
	  free(v->c);
	  v->c = NULL;
     }
     v->ncache = 0;
}

/***************************************************************************/

/* recursive loop over all cubature points for the given (m,mi) cache entry:
   add each point to the buffer buf, evaluating all at once whenever the
   buffer is full or when we are done */
static int compute_cacheval(const unsigned *m, unsigned mi, 
			    double *val, size_t *vali,
			    unsigned fdim, integrand_v f, void *fdata,
			    unsigned dim, unsigned id, double *p,
			    const double *xmin, const double *xmax,
			    double *buf, size_t nbuf, size_t *ibuf)
{
     if (id == dim) { /* add point to buffer of points */
	  memcpy(buf + (*ibuf)++ * dim, p, sizeof(double) * dim);
	  if (*ibuf == nbuf) { /* flush buffer */
	       if (f(dim, nbuf, buf, fdata, fdim, val + *vali))
		    return FAILURE;
	       *vali += *ibuf * fdim;
	       *ibuf = 0;
	  }
     }
     else {
	  double c = (xmin[id] + xmax[id]) * 0.5;
	  double r = (xmax[id] - xmin[id]) * 0.5;
	  const double *x = clencurt_x 
	       + ((id == mi) ? (m[id] ? (1 << (m[id] - 1)) : 0) : 0);
	  unsigned i, nx = (id == mi ? (m[id] ? (1 << (m[id] - 1)) : 1)
			    : (1 << (m[id])));
	  if (id != mi) {
	       p[id] = c;
	       if (compute_cacheval(m, mi, val, vali, fdim, f, fdata,
				    dim, id + 1, p,
				    xmin, xmax, buf, nbuf, ibuf))
		    return FAILURE;
	  }
	  for (i = 0; i < nx; ++i) {
	       p[id] = c + r * x[i];
	       if (compute_cacheval(m, mi, val, vali, fdim, f, fdata,
				    dim, id + 1, p,
				    xmin, xmax, buf, nbuf, ibuf))
		    return FAILURE;
	       p[id] = c - r * x[i];
	       if (compute_cacheval(m, mi, val, vali, fdim, f, fdata,
				    dim, id + 1, p,
				    xmin, xmax, buf, nbuf, ibuf))
		    return FAILURE;
	  }
     }
     return SUCCESS;
}

static size_t num_cacheval(const unsigned *m, unsigned mi, unsigned dim)
{
     unsigned i;
     size_t nval = 1;
     for (i = 0; i < dim; ++i) {
	  if (i == mi)
	       nval *= m[i] == 0 ? 2 : (1 << (m[i]));
	  else
	       nval *= (1 << (m[i] + 1)) + 1;
     }
     return nval;
}

static int add_cacheval(valcache *vc,
			const unsigned *m, unsigned mi,
			unsigned fdim, integrand_v f, void *fdata,
			unsigned dim, const double *xmin, const double *xmax,
			double *buf, size_t nbuf)
{
     size_t ic = vc->ncache;
     size_t nval, vali = 0, ibuf = 0;
     double p[MAXDIM];

     vc->c = (cacheval *) realloc(vc->c, sizeof(cacheval) * ++(vc->ncache));
     if (!vc->c) return -1;

     vc->c[ic].mi = mi;
     memcpy(vc->c[ic].m, m, sizeof(unsigned) * dim);
     nval = fdim * num_cacheval(m, mi, dim);
     vc->c[ic].val = (double *) malloc(sizeof(double) * nval);
     if (!vc->c[ic].val) return FAILURE;

     if (compute_cacheval(m, mi, vc->c[ic].val, &vali,
			  fdim, f, fdata,
			  dim, 0, p, xmin, xmax,
			  buf, nbuf, &ibuf))
	  return FAILURE;

     if (ibuf > 0) /* flush remaining buffer */
	  return f(dim, ibuf, buf, fdata, fdim, vc->c[ic].val + vali);

     return SUCCESS;
}

/***************************************************************************/

/* recursive loop to evaluate the integral contribution from the cache
   entry c, accumulating in val, for the given m[] except with m[md]
   -> m[md] - 1 if md < dim, using the cached values (cm,cmi,cval).  id is the
   current loop dimension (from 0 to dim-1). */
static unsigned eval(const unsigned *cm, unsigned cmi, double *cval,
		 const unsigned *m, unsigned md,
		 unsigned fdim, unsigned dim, unsigned id,
		 double weight, double *val)
{
     size_t voff = 0; /* amount caller should offset cval array afterwards */
     if (id == dim) {
	  unsigned i;
	  for (i = 0; i < fdim; ++i) val[i] += cval[i] * weight;
	  voff = fdim;
     }
     else if (m[id] == 0 && id == md) /* using trivial rule for this dim */ {
	  voff = eval(cm, cmi, cval, m, md, fdim, dim, id+1, weight*2, val);
	  voff += fdim * (1 << cm[id]) * 2
	       * num_cacheval(cm + id+1, cmi - (id+1), dim - (id+1));
     }
     else {
	  unsigned i;
	  unsigned mid = m[id] - (id == md); /* order of C-C rule */
	  const double *w = clencurt_w + mid + (1 << mid) - 1
	       + (id == cmi ? (cm[id] ? 1 + (1 << (cm[id]-1)) : 1) : 0);
	  unsigned cnx = (id == cmi ? (cm[id] ? (1 << (cm[id]-1)) : 1)
			  : (1 << (cm[id])));
	  unsigned nx = cm[id] <= mid ? cnx : (1 << mid);

	  if (id != cmi) {
	       voff = eval(cm, cmi, cval, m, md, fdim, dim, id + 1,
			   weight * w[0], val);
	       ++w;
	  }
	  for (i = 0; i < nx; ++i) {
	       voff += eval(cm, cmi, cval + voff, m, md, fdim, dim, id + 1,
			    weight * w[i], val);
	       voff += eval(cm, cmi, cval + voff, m, md, fdim, dim, id + 1,
			    weight * w[i], val);
	  }

	  voff += (cnx - nx) * fdim * 2
	       * num_cacheval(cm + id+1, cmi - (id+1), dim - (id+1));
     }
     return voff;
}

/* loop over all cache entries that contribute to the integral,
   (with m[md] decremented by 1) */
static void evals(valcache vc, const unsigned *m, unsigned md,
		  unsigned fdim, unsigned dim, 
		  double V, double *val)
{
     size_t i;

     memset(val, 0, sizeof(double) * fdim);
     for (i = 0; i < vc.ncache; ++i) {
	  if (vc.c[i].mi >= dim ||
	      vc.c[i].m[vc.c[i].mi] + (vc.c[i].mi == md) <= m[vc.c[i].mi])
	       eval(vc.c[i].m, vc.c[i].mi, vc.c[i].val,
		    m, md, fdim, dim, 0, V, val);
     }
}

/* evaluate the integrals for the given m[] using the cached values in vc,
   storing the integrals in val[], the error estimate in err[], and the
   dimension to subdivide next (the largest error contribution) in *mi */
static void eval_integral(valcache vc, const unsigned *m, 
			  unsigned fdim, unsigned dim, double V,
			  unsigned *mi, double *val, double *err, double *val1)
{
     double maxerr = 0;
     unsigned i, j;
     
     evals(vc, m, dim, fdim, dim, V, val);

     /* error estimates along each dimension by comparing val with
	lower-order rule in that dimension; overall (conservative)
	error estimate from maximum error of lower-order rules. */
     memset(err, 0, sizeof(double) * fdim);
     *mi = 0;
     for (i = 0; i < dim; ++i) {
	  double emax = 0;
	  evals(vc, m, i, fdim, dim, V, val1);
	  for (j = 0; j < fdim; ++j) {
	       double e = fabs(val[j] - val1[j]);
	       if (e > emax) emax = e;
	       if (e > err[j]) err[j] = e;
	  }
	  if (emax > maxerr) {
	       maxerr = emax;
	       *mi = i;
	  }
     }
     /* printf("eval: %g +/- %g (dim %u)\n", val[0], err[0], *mi); */
}

/***************************************************************************/

static int pconverged(unsigned fdim, const double *vals, const double *errs,
		     double reqAbsError, double reqRelError, error_norm norm)
#define ERR(j) errs[j]
#define VAL(j) vals[j]
#include "converged.h"

/***************************************************************************/
/* Vectorized version with user-supplied buffer to store points and values.
   The buffer *buf should be of length *nbuf * dim on entry (these parameters
   are changed upon return to the final buffer and length that was used).
   The buffer length will be kept <= max(max_nbuf, 1) * dim.

   Also allows the caller to specify an array m[dim] of starting degrees
   for the rule, which upon return will hold the final degrees.  The
   number of points in each dimension i is 2^(m[i]+1) + 1. */
   
int pcubature_v_buf(unsigned fdim, integrand_v f, void *fdata,
		    unsigned dim, const double *xmin, const double *xmax,
		    size_t maxEval,
		    double reqAbsError, double reqRelError,
		    error_norm norm,
		    unsigned *m,
		    double **buf, size_t *nbuf, size_t max_nbuf,
		    double *val, double *err)
{
     int ret = FAILURE;
     double V = 1;
     size_t numEval = 0, new_nbuf;
     unsigned i;
     valcache vc = {0, NULL};
     double *val1 = NULL;

     if (fdim <= 1) norm = ERROR_INDIVIDUAL; /* norm is irrelevant */
     if (norm < 0 || norm > ERROR_LINF) return FAILURE; /* invalid norm */

     if (fdim == 0) return SUCCESS; /* nothing to do */
     if (dim > MAXDIM) return FAILURE; /* unsupported */
     if (dim == 0) { /* trivial case */
	  if (f(0, 1, xmin, fdata, fdim, val)) return FAILURE;
          for (i = 0; i < fdim; ++i) err[i] = 0;
          return SUCCESS;
     }

     for (i = 0; i < fdim; ++i) {
	  val[i] = 0;
	  err[i] = HUGE_VAL;
     }

     for (i = 0; i < dim; ++i)
	  V *= (xmax[i] - xmin[i]) * 0.5; /* scale factor for C-C volume */

     new_nbuf = num_cacheval(m, dim, dim);

     if (max_nbuf < 1) max_nbuf = 1;
     if (new_nbuf > max_nbuf) new_nbuf = max_nbuf;
     if (*nbuf < new_nbuf) {
	  free(*buf);
	  *buf = (double *) malloc(sizeof(double) 
				   * (*nbuf = new_nbuf) * dim);
	  if (!*buf) goto done;
     }

     /* start by evaluating the m=0 cubature rule */
     if (add_cacheval(&vc, m, dim, fdim, f, fdata, dim, xmin, xmax, 
		       *buf, *nbuf) != SUCCESS)
	  goto done;

     val1 = (double *) malloc(sizeof(double) * fdim);

     while (1) {
	  unsigned mi;

	  eval_integral(vc, m, fdim, dim, V, &mi, val, err, val1);
	  if (pconverged(fdim, val, err, reqAbsError, reqRelError, norm)
	      || (numEval > maxEval && maxEval)) {
	       ret = SUCCESS;
	       goto done;
	  }
	  m[mi] += 1;
	  if (m[mi] > clencurt_M) goto done; /* FAILURE */

	  new_nbuf = num_cacheval(m, mi, dim);
	  if (new_nbuf > *nbuf && *nbuf < max_nbuf) {
	       *nbuf = new_nbuf;
	       if (*nbuf > max_nbuf) *nbuf = max_nbuf;
	       free(*buf);
	       *buf = (double *) malloc(sizeof(double) * *nbuf * dim);
	       if (!*buf) goto done; /* FAILURE */
	  }

	  if (add_cacheval(&vc, m, mi, fdim, f, fdata, 
			   dim, xmin, xmax, *buf, *nbuf) != SUCCESS)
	       goto done; /* FAILURE */
	  numEval += new_nbuf;
     }

done:
     free(val1);
     free_cachevals(&vc);
     return ret;
}

/***************************************************************************/

#define DEFAULT_MAX_NBUF (1U << 30) // original is 1U << 20

int pcubature_v(unsigned fdim, integrand_v f, void *fdata,
		unsigned dim, const double *xmin, const double *xmax,
		size_t maxEval, double reqAbsError, double reqRelError,
		error_norm norm,
		double *val, double *err)
{
     int ret;
     size_t nbuf = 0;
     unsigned m[MAXDIM];
     double *buf = NULL;
     memset(m, 0, sizeof(unsigned) * dim);
     ret = pcubature_v_buf(fdim, f, fdata, dim, xmin, xmax,
				  maxEval, reqAbsError, reqRelError, norm,
				  m, &buf, &nbuf, DEFAULT_MAX_NBUF, val, err);
     free(buf);
     return ret;
}

#include "vwrapper.h"

int pcubature(unsigned fdim, integrand f, void *fdata,
	      unsigned dim, const double *xmin, const double *xmax,
	      size_t maxEval, double reqAbsError, double reqRelError,
	      error_norm norm,
	      double *val, double *err)
{
     int ret;
     size_t nbuf = 0;
     unsigned m[MAXDIM];
     double *buf = NULL;
     fv_data d;

     d.f = f; d.fdata = fdata;
     memset(m, 0, sizeof(unsigned) * dim);
     ret = pcubature_v_buf(
	  fdim, fv, &d, dim, xmin, xmax, 
	  maxEval, reqAbsError, reqRelError, norm,
	  m, &buf, &nbuf, 16 /* max_nbuf > 0 to amortize function overhead */,
	  val, err);
     free(buf);
     return ret;
}
