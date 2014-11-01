
ctypedef enum error_norm: ERROR_INDIVIDUAL = 0, ERROR_PAIRED, ERROR_L2,\
                         ERROR_L1, ERROR_LINF

ctypedef int (*integrand) (unsigned ndim, const double *x, void *fdata, unsigned fdim,
                           double *fval)

ctypedef int (*integrand_v) (unsigned ndim, size_t npt, const double *x,
                             void *fdata, unsigned fdim, double *fval)

cdef extern from './cpackage/cubature.h':
    int hcubature(unsigned fdim, integrand f, void *fdata,
                  unsigned ndim, const double *xmin, const double *xmax,
                  unsigned maxEval, double reqAbsError, double reqRelError,
                  error_norm norm, double *val, double *err)

    int pcubature(unsigned fdim, integrand f, void *fdata,
                  unsigned ndim, const double *xmin, const double *xmax,
                  size_t maxEval, double reqAbsError, double reqRelError,
                  error_norm norm, double *val, double *err)

    int hcubature_v(unsigned fdim, integrand_v f, void *fdata,
                    unsigned ndim, const double *xmin, const double *xmax,
                    size_t maxEval, double reqAbsError, double reqRelError,
                    error_norm norm, double *val, double *err)

    int pcubature_v(unsigned fdim, integrand_v f, void *fdata,
                    unsigned ndim, const double *xmin, const double *xmax,
                    size_t maxEval, double reqAbsError, double reqRelError,
                    error_norm norm, double *val, double *err)
# Vectorized version with user-supplied buffer to store points and values.
# The buffer *buf should be of length *nbuf * dim on entry (these parameters
# are changed upon return to the final buffer and length that was used).
# The buffer length will be kept <= max(max_nbuf, 1) * dim.
#
# Also allows the caller to specify an array m[dim] of starting degrees
# for the rule, which upon return will hold the final degrees.  The
# number of points in each dimension i is 2^(m[i]+1) + 1.
    int pcubature_v_buf(unsigned fdim, integrand_v f, void *fdata,
                    unsigned ndim, const double *xmin, const double *xmax,
                    size_t maxEval, double reqAbsError, double reqRelError,
                    error_norm norm, unsigned *m,
                    double **buf, size_t *nbuf, size_t max_nbuf,
                    double *val, double *err)

#TODO these guys are needed as explained in Cython's "tricks an tips"
cdef extern from *:
    pass

cdef extern from "./cpackage/hcubature.c":
    #TODO this guy should not be here... but was needed to
    #     avoid a link error at the end of the compilation
    pass

cdef extern from "./cpackage/pcubature.c":
    #TODO this guy should not be here... but was needed to
    #     avoid a link error at the end of the compilation
    pass
