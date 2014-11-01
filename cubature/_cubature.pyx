cimport numpy as np
import numpy as np
import cython
from cpython cimport tuple, bool
from _cubature cimport error_norm, integrand, integrand_v,\
                      hcubature, pcubature, hcubature_v, pcubature_v

DOUBLE = np.float64
ctypedef np.double_t cDOUBLE

cdef object f
cdef np.ndarray x_buffer

#TODO list
# - try to avoid the x_buffer and fval_buffer
# - generalize and test the case when a Cython function defined by:
#   cdef AND cpdef is passed as the integrand function
# integrand_cb means integrand_callback, one way to pass the Python function
# to the C code
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int integrand_cb(unsigned ndim, double *x, void *fdata,
                      unsigned fdim, double *fval):
    global f, x_buffer
    cdef np.ndarray[cDOUBLE, ndim=1] fval_buffer
    cdef unsigned j, k
    for j in range(ndim):
        x_buffer[j] = x[j]
    fval_buffer = (<object>f)(x_buffer, *<tuple>fdata)
    for k in range(fdim):
        fval[k] = fval_buffer[k]
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int integrand_v_cb(unsigned ndim, size_t npt, double *x, void *fdata,
                        unsigned fdim, double *fval):
    global f
    cdef np.ndarray[cDOUBLE, ndim=1] fval_buffer, x_buffer
    cdef unsigned i
    x_buffer = np.empty(ndim*npt, dtype=DOUBLE)
    for i in range(npt*ndim):
        x_buffer[i] = x[i]
    fval_buffer = (<object>f)(x_buffer, npt, *<tuple>fdata)
    for i in range(npt*fdim):
        fval[i] = fval_buffer[i]
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
def _cubature(pythonf,
              unsigned fdim,
              np.ndarray[cDOUBLE, ndim=1] xmin,
              np.ndarray[cDOUBLE, ndim=1] xmax,
              fdata,
              str adaptive,
              double abserr, double relerr, int norm,
              unsigned maxEval,
              bool vectorized,
              np.ndarray[cDOUBLE, ndim=1] val,
              np.ndarray[cDOUBLE, ndim=1] err,
             ):

    global f, x_buffer
    cdef int ans
    cdef unsigned ndim
    ndim = xmin.shape[0]
    x_buffer = np.empty((ndim), dtype=DOUBLE)
    f  = pythonf
    if adaptive == 'h':
        if vectorized:
            ans =  hcubature_v(fdim,
                               <integrand_v> integrand_v_cb,
                               <void *> fdata,
                               ndim,
                               <double *> xmin.data,
                               <double *> xmax.data,
                               maxEval,
                               abserr,
                               relerr,
                               <error_norm> norm,
                               <double *> val.data,
                               <double *> err.data)
        else:
            ans =  hcubature(fdim,
                             <integrand> integrand_cb,
                             <void *> fdata,
                             ndim,
                             <double *> xmin.data,
                             <double *> xmax.data,
                             maxEval,
                             abserr,
                             relerr,
                             <error_norm> norm,
                             <double *> val.data,
                             <double *> err.data)
    else:
        if vectorized:
            ans =  pcubature_v(fdim,
                               <integrand_v> integrand_v_cb,
                               <void *> fdata,
                               ndim,
                               <double *> xmin.data,
                               <double *> xmax.data,
                               maxEval,
                               abserr,
                               relerr,
                               <error_norm> norm,
                               <double *> val.data,
                               <double *> err.data)
        else:
            ans =  pcubature(fdim,
                             <integrand> integrand_cb,
                             <void *> fdata,
                             ndim,
                             <double *> xmin.data,
                             <double *> xmax.data,
                             maxEval,
                             abserr,
                             relerr,
                             <error_norm> norm,
                             <double *> val.data,
                             <double *> err.data)
    return ans
