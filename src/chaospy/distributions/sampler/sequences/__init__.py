"""
In mathematics, a `low-discrepancy sequence`_ is a sequence with the property
that for all values of N, its subsequence x1, ..., xN has a low discrepancy.

Roughly speaking, the discrepancy of a sequence is low if the proportion of
points in the sequence falling into an arbitrary set B is close to proportional
to the measure of B, as would happen on average (but not for particular
samples) in the case of an equi-distributed sequence. Specific definitions of
discrepancy differ regarding the choice of B (hyperspheres, hypercubes, etc.)
and how the discrepancy for every B is computed (usually normalized) and
combined (usually by taking the worst value).

Low-discrepancy sequences are also called quasi-random or sub-random sequences,
due to their common use as a replacement of uniformly distributed random
numbers. The "quasi" modifier is used to denote more clearly that the values of
a low-discrepancy sequence are neither random nor pseudorandom, but such
sequences share some properties of random variables and in certain applications
such as the quasi-Monte Carlo method their lower discrepancy is an important
advantage.

.. low-discrepancy sequence: https://en.wikipedia.org/wiki/Low-discrepancy_sequence
"""
from .chebyshev import create_chebyshev_samples, create_nested_chebyshev_samples
from .grid import create_grid_samples, create_nested_grid_samples
from .halton import create_halton_samples
from .hammersley import create_hammersley_samples
from .sobol import create_sobol_samples
from .korobov import create_korobov_samples
