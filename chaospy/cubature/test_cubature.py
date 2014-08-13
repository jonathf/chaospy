import sys
import os
from subprocess import Popen, PIPE
from operator import mod

import numpy as np
from numpy import pi, sin, cos, exp, zeros, ones, float64

from cubature import cubature

count = 0
fdim = 1
function = 0
ndim = 0
K_2_SQRTPI = 1.12837916709551257390
radius = 0.50124145262344534123412

class Logger(object):
    def __init__(self):
        self.file = open('test_cubature.txt', 'w')
        self.file.close()
        self.stdout = sys.stdout
        sys.stdout = self
    def write(self, text):
        self.file = open('test_cubature.txt', 'a')
        self.file.write(text)
        self.file.close()
        self.stdout.write(text)

def f_test(x, *args):
    global count, ndim, fdim, function, K_2_SQRTPI, radius
    count += 1
    if function == 0:
        val = ones(fdim, float64)
        for i in range(ndim):
            val *= cos(x[i])
    elif function == 1:
        scale = 1.
        val = zeros(fdim, float64)
        for i in range(ndim):
            if x[i] > 0:
                z = (1 - x[i]) / x[i]
                val += z**2
                scale *= K_2_SQRTPI / x[i]**2
            else:
                scale = 0
        val = scale * exp(-val)
    elif function == 2:
        val = zeros(fdim, float64)
        for i in range(ndim):
            val += x[i]**2
        val[:] = (val[0] < radius**2)
    elif function == 3:
        val = ones(fdim, float64)
        for i in range(ndim):
            val *= 2.0 * x[i]
    elif function == 4:
        sum1 = zeros(fdim, float64)
        sum2 = zeros(fdim, float64)
        a = 0.1
        for i in range(ndim):
            dx1 = x[i] - 1. / 3.
            dx2 = x[i] - 2. / 3.
            sum1 += dx1**2
            sum2 += dx2**2
        return 0.5 * pow (K_2_SQRTPI / (2. * a), ndim) \
                   * (exp (-sum1 / a**2) + exp (-sum2 / a**2))
    elif function == 5:
        sum1 = zeros(fdim, float64)
        sum2 = zeros(fdim, float64)
        a = 0.1
        for i in range(ndim):
            dx1 = x[i] - 1. / 3.
            dx2 = x[i] - 2. / 3.
            sum1 += dx1 * dx1
            sum2 += dx2 * dx2
        return 0.5 * pow (K_2_SQRTPI / (2. * a), ndim) \
                   * (exp (-sum1 / a**2) + exp (-sum2 / a**2))
    elif function == 6:
        val = ones(fdim, float64)
        c = (1.+ 10.**0.5)/9.
        for i in range(ndim):
            val *= c / (c + 1) * pow((c + 1) / (c + x[i]), 2.0)
    elif function == 7:
        p = ones(fdim, float64) / ndim
        val = pow(1 + p, ndim)
        for i in range(ndim):
            val *= pow(x[i], p)
    return val

def f_test_vec(xvec, npt, *args):
    global ndim, fdim
    out = np.zeros(fdim*npt)
    x = np.zeros(ndim)
    for i in range(npt):
        for j in range(ndim):
            x[j] = xvec[i*ndim+j]
        out[i*fdim:(i+1)*fdim] = f_test(x, *args)
    return out


def exact0(ndim, xmax):
    val = 1.
    for i in range(ndim):
        val *= sin(xmax[i])
    return val

def exact2_S(n):
     fact = 1
     if mod(n, 2) == 0:
         val = 2 * pow(pi, n * 0.5)
         n = n / 2
         while (n > 1):
             fact *= n
             n -= 1
         val /= fact
     else:
         val = (1 << (n/2 + 1)) * pow(pi, n/2)
         while (n > 2):
             fact *= n
             n -= 2
         val /= fact
     return val

def exact2(ndim, xmax):
    global radius
    val = 1 if ndim == 0 else exact2_S(ndim) * pow(radius * 0.5, ndim) / ndim
    return val

def exact_integral(i, ndim, xmax):
    if i in [1,3,4,5,6,7]:
        return 1.
    elif i == 0:
        return exact0(ndim, xmax)
    elif i == 2:
        return exact2(ndim, xmax)

def main(lndim, tol, functions, maxEval, lfdim):
    global count, ndim, fdim, function
    ndim = lndim
    fdim = lfdim
    logger = Logger() # instanciate Logger to redirect print to test_cubature.txt
    xmin = zeros(ndim)
    xmax = ones(ndim)
    abserr = tol
    relerr = tol

    for vectorized in [False, True]:
        print('======================================')
        print('           VECTORIZED={0}'.format(vectorized))
        print('======================================')
        print('')
        for function in functions:
            count = 0
            print('______________________________________')
            print('                                      ')
            print('                CASE {0}'.format(function))
            print('______________________________________')
            for adaptive in ['h', 'p']:
                if adaptive == 'h':
                    print('__________________')
                    print('                  ')
                    print('Testing h_adaptive')
                    print('__________________')
                else:
                    print('__________________')
                    print('                  ')
                    print('Testing p_adaptive')
                    print('__________________')
                print('')
                print('Python Cubature:')
                print('----------------')
                print('{0}-dim integral, tolerance = {1}'.format(ndim,
                                                                 relerr))
                print('')
                if vectorized:
                    val, err = cubature(ndim, f_test_vec, xmin, xmax, (),
                             adaptive, abserr, relerr, norm=0,
                             maxEval=maxEval, vectorized=vectorized)
                else:
                    val, err = cubature(ndim, f_test, xmin, xmax, (),
                             adaptive, abserr, relerr, norm=0,
                             maxEval=maxEval, vectorized=vectorized)
                true_err = abs(val[0] - exact_integral(function, ndim, xmax))
                print('integrand {0}: integral = {1}, est err = {2}, true err = {3:e}'
                      .format(function, val[0], err[0], true_err))
                print('')
                print('#evals = {0}'.format(count))
                print('')

                htest_path = os.path.join('.', 'cpackage', 'htest.exe')
                ptest_path = os.path.join('.', 'cpackage', 'ptest.exe')
                if os.path.isfile(htest_path) or os.path.isfile(ptest_path):
                    print('C Cubature:')
                    print('-----------')
                else:
                    print('C Cubature program not compiled!')
                    print('compile using (more details in ".\cpackage\README:"')
                    print('\tcc -DHCUBATURE -o htest hcubature.c test.c -lm')
                    print('\tcc -DPCUBATURE -o ptest pcubature.c test.c -lm')
                fdim_str = '/'.join(['x' for j in range(fdim)])
                if adaptive=='h' and os.path.isfile(htest_path):
                    p = Popen([htest_path] +
                          list(map(str, [ndim,tol,function,
                                         maxEval,fdim_str])),
                          stdout = PIPE)
                    p.wait()
                    for l in p.stdout: print(l)
                if adaptive=='p' and os.path.isfile(ptest_path):
                    p = Popen([ptest_path] +
                          list(map(str, [ndim,tol,function,
                                         maxEval,fdim_str])),
                          stdout = PIPE)
                    p.wait()
                    for l in p.stdout: print(l)

if __name__ == '__main__':
    from __init__ import run_test
    run_test()
