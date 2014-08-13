/*
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

/* This stand-alone program, which should be compiled and linked against
   FFTW (www.fftw.org) version 3 or later, is used to generate the clencurt.h
   file for pcubature.c.  You only need to run it if you want to do
   p-adaptive cubature with more than 8193 points per dimension.  See
   the README file for more information. */


#include <stdlib.h>
#include <stdio.h>
#include <fftw3.h>

extern long double cosl(long double x);

/* Program to generate tables of precomputed points and weights
   for Clenshaw-Curtis quadrature on an interval [-1, 1] with

        3, 5, 9, ..., 2^(m+1)+1, ..., 2^(M+1)+1

   points up to some given M.  Because the quadrature rules are
   mirror-symmetric, we only need to store 2^m+1 weights for each rule.

   Furthermore, the rules are nested, so we only need to store the
   points for the M rule and the points for the other rules are a subset.
   We store the points and weights in a permuted order P corresponding to a
   usage where we first evaluate m=0, then m=1, etc. until it is converged.

   In particular, for the m rule (2^m+1 weights w[j], j=0,1,...,2^m),
   the corresponding points are

       x[j] = +/- cos(pi * j / 2^(m+1))

   (Note that for x[2^m] = 0; this point must be specially handled
    so that it is not counted twice.)

   So, we precompute an array clencurt_x of length 2^M storing

       clencurt_x[j] = cos(pi * P_M(j) / 2^(M+1))

   for j = 0,1,...,2^M-1.  Then, for a given rule m, we use

      x[P_m(j)] = clencurt_x[j]

   for j = 0,1,...,2^m-1 and x=0 for j = 2^m.  P_m is the permutation
 
      P_0(j) = j
      P_m(j) = 2 * P_{m-1}(j)          if j < 2^(m-1)
               2 * (j - 2^(m-1)) + 1   otherwise 

   The 2^m+1 weights w are stored for m=0,1,..,M in the same order in an array
   clencurt_w of length M+2^(M+1), in order of m.  So, the weights for
   a given m start at clencurt_w + (m + 2^m - 1), in the same order as
   clencurt_x except that it starts with the weight for x=0.
*/

static int P(int m, int j)
{
     if (m == 0) return j;
     else if (j < (1<<(m-1))) return 2 * P(m-1,j);
     else return 2 * (j - (1<<(m-1))) + 1;
}

/***************************************************************************/
/* The general principle is this: in Fejer and Clenshaw-Curtis quadrature,
   we take our function f(x) evaluated at f(cos(theta)), discretize
   it in theta to a vector f of points, compute the DCT, then multiply
   by some coefficients c (the integrals of cos(theta) * sin(theta)).
   In matrix form, given the DCT matrix D, this is:

             c' * D * f = (D' * c)' * f = w' * f

   where w is the vector of weights for each function value.  It
   is obviously much nicer to precompute w if we are going to be
   integrating many functions.   Since w = D' * c, and the transpose
   D' of a DCT is another DCT (e.g. the transpose of DCT-I is a DCT-I,
   modulo some rescaling in FFTW's normalization), to compute w is just
   another DCT.

   There is an additional wrinkle, because the odd-frequency DCT components
   of f integrate to zero, so every other entry in c is zero.  We can
   take advantage of this in computing w, because we can essentially do
   a radix-2 step in the DCT where one of the two subtransforms is zero.
   Therefore, for 2n+1 inputs, we only need to do a DCT of size n+1, and
   the weights w are a nice symmetric function.

   The weights are for integration of functions on (-1,1).
*/

void clencurt_weights(int n, long double *w)
{
     fftwl_plan p;
     int j;
     long double scale = 1.0 / n;
     
     p = fftwl_plan_r2r_1d(n+1, w, w, FFTW_REDFT00, FFTW_ESTIMATE);
     for (j = 0; j <= n; ++j) w[j] = scale / (1 - 4*j*j);
     fftwl_execute(p);
     w[0] *= 0.5;
     fftwl_destroy_plan(p);
}
/***************************************************************************/

#define KPI 3.1415926535897932384626433832795028841971L

int main(int argc, char **argv)
{
     int M = argc > 1 ? atoi(argv[1]) : 11;
     long double *w;
     int j, m;
     long double k;
     
     if (argc > 2 || M < 0) {
	  fprintf(stderr, "clencurt_gen usage: clencurt_gen [M]\n");
	  return EXIT_FAILURE;
     }

     w = (long double *) fftwl_malloc(sizeof(long double) * ((1<<(M+2)) - 2));
     if (!w) {
	  fprintf(stderr, "clencurt_gen: out of memory\n");
	  return EXIT_FAILURE;
     }	  

     printf("/* AUTOMATICALLY GENERATED -- DO NOT EDIT */\n\n");
 
     printf("static const int clencurt_M = %d;\n\n", M);

     printf("static const double clencurt_x[%d] = { /* length 2^M */\n", 1<<M);
     k = KPI / ((long double) (1<<(M+1)));
     for (j = 0; j < (1<<M); ++j)
	  printf("%0.18Lg%s\n", cosl(k*P(M,j)), j == (1<<M)-1 ? "" : ",");
     printf("};\n\n");

     printf("static const double clencurt_w[%d] = { /* length M+2^(M+1) */\n",
	    M + (1<<(M+1)));
     for (m = 0; m <= M; ++m) {
	  clencurt_weights(1 << m, w);
	  printf("/* m = %d: */ %0.18Lg,\n", m, w[1 << m]);
	  for (j = 0; j < (1 << m); ++j)
	       printf("%0.18Lg%s\n", w[P(m,j)], 
		      j == (1<<m)-1 && m == M ? "" : ",");
     }
     printf("};\n");

     printf("\n/* P_M =");
     for (j = 0; j < (1<<M); ++j)
	  printf(" %d", P(M,j));
     printf(" */\n");
     
     return EXIT_SUCCESS;
}
