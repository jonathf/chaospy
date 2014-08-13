import os
import shutil
import subprocess

from cubature import cubature
from test_cubature import main

__version__ = '0.12.0'

def run_test(ndim=3, tol=1.e-5, functions=[0,1,2,3,4,5,6,7],
             maxEval=1000000, fdim=5):
    '''Default test for the Cubature package.

    Parameters
    ----------

    ndim : integer
        Number of dimensions to integrate over.
        (default = 3)

    tol : float
        Error tolerance for the test
        (default = 1.e-5)

    functions : list or tuple
        (default = [0,1,2,3,4,5,6,7])
        The different test functions are:
        0 - a product of cosine functions
        1 - a Gaussian integral of exp(-x^2), remapped to [0,infinity) limits
        2 - volume of a hypersphere (integrating a discontinuous function!)
        3 - a simple polynomial (product of coordinates)
        4 - a Gaussian centered in the middle of the integration volume
        5 - a sum of two Gaussians
        6 - an example function by Tsuda, a product of terms with near poles
        7 - a test integrand by Morokoff and Caflisch, a simple product of
           dim-th roots of the coordinates (weakly singular at the boundary)
    maxEval : integer
        Maximum number of function evaluations.
        (default = 1000000)
    fdim : integer
        Size for the vector-valued function.
        (default = 5)
    '''
    cwd = os.getcwd()
    main(ndim, tol, functions, maxEval, fdim)
