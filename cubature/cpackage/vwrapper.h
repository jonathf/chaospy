/* vectorized wrapper around non-vectorized integrands */
#ifndef VWRAPPER_H
#define VWRAPPER_H
typedef struct fv_data_s { integrand f; void *fdata; } fv_data;
static int fv(unsigned ndim, size_t npt,
	      const double *x, void *d_,
	      unsigned fdim, double *fval)
{
     fv_data *d = (fv_data *) d_;
     integrand f = d->f;
     void *fdata = d->fdata;
     unsigned i;
     /* printf("npt = %u\n", npt); */
     for (i = 0; i < npt; ++i) 
	  if (f(ndim, x + i*ndim, fdata, fdim, fval + i*fdim))
	       return FAILURE;
     return SUCCESS;
}
#endif /* VWRAPPER_H */
