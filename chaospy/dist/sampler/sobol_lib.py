"""
Generates quasirandom Sobol vector.

This code is distributed under the GNU LGPL license.

The routine adapts the ideas of Antonov and Saleev.
Original FORTRAN77 version by Bennett Fox.
MATLAB version by John Burkardt.
PYTHON version by Corrado Chisari

Reference:

      Antonov, Saleev,
      USSR Computational Mathematics and Mathematical Physics,
      Volume 19, 1980, pages 252 - 256.

      Paul Bratley, Bennett Fox,
      Algorithm 659:
      Implementing Sobol's Quasirandom Sequence Generator,
      ACM Transactions on Mathematical Software,
      Volume 14, Number 1, pages 88-100, 1988.

      Bennett Fox,
      Algorithm 647:
      Implementation and Relative Efficiency of Quasirandom 
      Sequence Generators,
      ACM Transactions on Mathematical Software,
      Volume 12, Number 4, pages 362-376, 1986.

      Ilya Sobol,
      USSR Computational Mathematics and Mathematical Physics,
      Volume 16, pages 236-242, 1977.

      Ilya Sobol, Levitan, 
      The Production of Points Uniformly Distributed in a Multidimensional 
      Cube (in Russian),
      Preprint IPM Akad. Nauk SSSR, 
      Number 40, Moscow 1976.
"""

import numpy as np
__author__ = "Jonathan Feinberg"
__email__ = "jonathan@feinberg.no"
__date__ = "2013-05-28"

_seed = 1

def bit_hi(n):
    bit = 0
    while True:
        if n <= 0:
            break
        bit += 1
        n = int(n/2)
    return bit

def bit_lo(n):
    bit = 0
    while True:
        bit += 1
        i = int(n/2)
        if n==2*i: break
        n = i
    return bit


def sobol(dim_num, N, seed=None):
    """
Parameters
----------
dim_num : int
    Number of spacial dimensions.
    Must satisfy 0<dim_num<41
N : int
    Number of unique samples to generate
seed : int, optional
    Starting seed. Non-positive values are treated as 1.
    If omited, consequtive samples are used.

Returns
-------
quasi : np.ndarray
    Quasi-random vector with `shape=(dim_num, N)`.
    """

    global _seed

    dim_max = 40
    log_max = 30

    if seed is None:
        seed = _seed
    _seed += N

    v = np.zeros((dim_max,log_max), dtype=int)
    v[0:40,0] = 1

    v[2:40,1] = 1, 3, 1, 3, 1, 3, 3, 1, \
                3, 1, 3, 1, 3, 1, 1, 3, 1, 3, \
                1, 3, 1, 3, 3, 1, 3, 1, 3, 1, \
                3, 1, 1, 3, 1, 3, 1, 3, 1, 3

    v[3:40,2] = 7, 5, 1, 3, 3, 7, 5, \
                5, 7, 7, 1, 3, 3, 7, 5, 1, 1, \
                5, 3, 3, 1, 7, 5, 1, 3, 3, 7, \
                5, 1, 1, 5, 7, 7, 5, 1, 3, 3

    v[5:40,3] = 1, 7, 9,13,11, \
                1, 3, 7, 9, 5,13,13,11, 3,15, \
                5, 3,15, 7, 9,13, 9, 1,11, 7, \
                5,15, 1,15,11, 5, 3, 1, 7, 9

    v[7:40,4] = 9, 3,27, \
               15,29,21,23,19,11,25, 7,13,17, \
                1,25,29, 3,31,11, 5,23,27,19, \
               21, 5, 1,17,13, 7,15, 9,31, 9

    v[13:40,5] = 37,33, 7, 5,11,39,63, \
                 27,17,15,23,29, 3,21,13,31,25, \
                  9,49,33,19,29,11,19,27,15,25

    v[19:40,6] = 13, \
                 33,115, 41, 79, 17, 29,119, 75, 73,105, \
                  7, 59, 65, 21,  3,113, 61, 89, 45,107

    v[37:40,7] = 7, 23, 39

    poly=    1,   3,   7,  11,  13,  19,  25,  37,  59,  47, \
            61,  55,  41,  67,  97,  91, 109, 103, 115, 131, \
           193, 137, 145, 143, 241, 157, 185, 167, 229, 171, \
           213, 191, 253, 203, 211, 239, 247, 285, 369, 299

    atmost = 2**log_max - 1

    #Find the number of bits in ATMOST.
    maxcol = bit_hi ( atmost )

    #Initialize row 1 of V.
    v[0,0:maxcol] = 1

    #Check parameters.
    assert 0 < dim_num < dim_max, "dim_num in [1,40]"

    #Initialize the remaining rows of V.
    for i in range(1 , dim_num):

        #The bits of the integer POLY(I) gives the form of polynomial I.
        #Find the degree of polynomial I from binary encoding.
        j = poly[i]
        m = 0
        while True:
            j = int(j/2)
            if j <= 0:
                break
            m += 1

        #Expand this bit pattern to separate components of the logical array INCLUD.
        j = poly[i]
        includ = np.zeros(m, dtype=bool)
        for k in range(m, 0, -1):
            j2 = int(j/2)
            includ[k-1] =  j != 2*j2
            j = j2

        #Calculate the remaining elements of row I as explained
        #in Bratley and Fox, section 2.
        for j in range(m+1, maxcol+1):
            newv = v[i, j-m-1].item()
            l = 1
            for k in range(1, m+1):
                l = 2 * l
                if includ[k-1]:
                    newv = newv ^ l*v[i,j-k-1].item()

            v[i,j-1] = newv

    v = v[:dim_num]


    #Multiply columns of V by appropriate power of 2.
    l = 1
    for j in range(maxcol-1, -1, -1):
        l = 2 * l
        v[:,j] = v[:,j] * l

    #RECIPD is 1/(common denominator of the elements in V).
    recipd = 1. / ( 2 * l )
    lastq=np.zeros(dim_num, dtype=int)

    seed = int(seed)
    if seed < 1:
        seed = 1

    for seed_ in range(seed):
        lastq[:] = lastq ^ v[:,bit_lo(seed_)-1]

    #Calculate the new components of QUASI.
    quasi=np.empty((dim_num, N))
    for j in range(N):
        quasi[:, j] = lastq * recipd
        lastq[:] = lastq ^ v[:,bit_lo(seed+j)-1]

    return quasi

