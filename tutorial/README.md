This directory contains various tutorials, presentations and examples.

Q & A
=====

Question 1
----------

Is it possible to create a orthonormal basis on an arbitrary custom inner
product definition?

Answer 1
--------

Yes, but it require that the algorithm be constructed.

For example, it is possible to implement a orthonormal Gram-Schmidt as follows:

```
V = list(cp.basis(0, polyOrder, dim))

P = [V[0]]

for v in V:
  for p in P:
    v = v - custom_inner(v, p) * p
  v = v / cp.sqrt( custom_inner(v, v) )
  P.append(v)

P = cp.aspolynomial(P)
```

Here `custom_inner` is the user provided inner product.

Alternative, assuming that the inner product defintion is decomposable, the
_three terms recursion_ algorithm or _discretized Stieltje's procedure_ can be
implemented.
Given that the recurrence cooeficients `A` and `B` are calculated, this can be
used to create orthonormal polynomials as follows:
```
x = cp.variable(dim)
orth = [ cp.basis(0, 0, dim), (x-A[:,0]) / np.sqrt(B[:,1]) ]
for n in range(1, order):
  orth.append( (orth[-1]*(x-A[:,n]) - orth[-2]*np.sqrt(B[:,n])) /
np.sqrt(B[:,n+1]) )
orth = cp.flatten( cp.aspolynomial(orth) )
```
