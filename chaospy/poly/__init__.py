"""
General tools for creating and manipulating polynomials

Classes
-------
Poly            General polynomial class

Unique Poly functions
---------------------
Hessian         Hessian differential operator
basis           Unit polynomial basis
call            Evaluate a polynomial along specified axes
cutoff          Remove polynomial components with a given order
decompose       Decompose a polynomial to component form
differential    Derivative along a given dimension
dimsplit        Split polynomial into dimensional components
gradient        Gradient differential operator
is_decomposed   Check if on component form
prange          Range of basic polynomials
setdim          Adjust the number of dimensions
substitute      Variable substitution
swapdim         Swap two given dimension
variable        Variable constructor

In addition there are a function that can be used on np.ndarray,
cp.frac and cp.Poly.

Overlapping functions
---------------------
all             Test if all values are true
any             Test if any values are true
asfloat         Convert quantity to float
asfrac          Convert quantity to frac
asint           Convert quantity to int
around          Evenly round to the given number of decimals
cumsum          Cumulative sum along a given axis
cumprod         Cumulative product along a given axis
diag            Extract or construct a diagonal
dtyping         Find the least common denomiator dtype
flatten         Flatten a quantity
inner           Inner product operator
mean            The arithmetic average
outer           Outer product operator
prod            Product of along a given axis
repeat          Repeat elements of a quantity
reshape         Give a new shape to a quantity
rollaxis        Roll the specified axis backwards
roll            Roll the dimensions of quantity
std             Sample standard deviation along a given axis
sum             Sum over a given axis
swapaxes        Interchange two axes of an quantity
toarray         Convert quantity to type ndarray
trace           Take the trace of a quantity
transpose       Transpose polynomial coefficients
tril            Lower triangle of quantity
var             Sample variance

"""

from base import *
from collection import *
from fraction import *
from wrapper import *

__version__ = "1.0"

