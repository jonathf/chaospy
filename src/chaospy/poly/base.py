"""
Polynomial base class Poly.
"""
import re
import numpy as np

import chaospy.poly


# Name of variables
VARNAME = "q"
POWER = "^"
SEP = ""


def setvarname(name):
    """Set the variable display name."""
    global VARNAME
    VARNAME = name


class Poly(object):
    """
    Polynomial representation in variable dimensions.

    Can also represent sets of polynomials.

    Examples:
        Direct construction:
        >>> P = cp.Poly({(1,):np.array(1)})
        >>> print(P)
        q0

        Basic operators:
        >>> x,y = cp.variable(2)
        >>> print(x**2 + x*y + 2)
        q0^2+q0q1+2

        Evaluation:
        >>> g = -3*x+x**2
        >>> print(g([1,2,3], [1,2,3]))
        [-2 -2  0]

        Arrays:
        >>> P = cp.Poly([x*y, x, y])
        >>> print(P)
        [q0q1, q0, q1]
    """

    __array_priority__ = 9000

    def __init__(self, core=None, dim=None, shape=None, dtype=None):
        """
        Constructor for the Poly class.

        Args:
            A (array_like, dict, Poly) : The polynomial coefficient Tensor.
                    Where A[(i,j,k)] corresponds to a_{ijk} x^i y^j z^k
                    (A[i][j][k] for list and tuple)
            dim (int) : the dimensionality of the polynomial.  Automatically
                    set if A contains a value.
            shape (tuple) : the number of polynomials represented.
                    Automatically set if A contains a value.
            dtype (type) : The type of the polynomial coefficients
        """
        core, dim, shape, dtype = chaospy.poly.constructor.preprocess(
            core, dim, shape, dtype)

        self.keys = sorted(core.keys(), key=sort_key)
        self.dim = dim
        self.shape = shape
        self.dtype = dtype
        self.A = core

    def __abs__(self):
        """Absolute value."""
        core = {key: abs(val) for key, val in self.A.items()}
        return Poly(core, self.dim, self.shape, self.dtype)

    def __add__(self, other):
        """Left addition."""
        return chaospy.poly.collection.arithmetics.add(self, other)

    def __radd__(self, other):
        """Right addition."""
        return chaospy.poly.collection.arithmetics.add(self, other)

    def __mul__(self, other):
        """Left multiplication."""
        return chaospy.poly.collection.arithmetics.mul(self, other)

    def __rmul__(self, other):
        """Right multiplication."""
        return chaospy.poly.collection.arithmetics.mul(self, other)

    def __sub__(self, other):
        """Left subtraction."""
        return chaospy.poly.collection.arithmetics.add(self, -other)

    def __rsub__(self, other):
        """Right subtraction."""
        return chaospy.poly.collection.arithmetics.add(-self, other)

    def __div__(self, other):
        """Python2 division."""
        return chaospy.poly.collection.arithmetics.mul(
            self, np.asfarray(other)**-1)
        return NotImplemented

    def __truediv__(self, other):
        """True division."""
        return self.__div__(other)

    def __floordiv__(self, other):
        return chaospy.poly.collection.asint(self.__div__(other))

    def __eq__(self, other):
        return ~self.__ne__(other)

    def __ne__(self, other):
        if not isinstance(other, Poly):
            other = Poly(other)
        diff = abs(self - other)
        out = np.zeros(diff.shape, dtype=bool)
        for key in diff.keys:
            out = out + (diff.A[key]!=0)
        return out

    def __call__(self, *args, **kws):
        """
        Evaluate a polynomial.

        Args:
            args (array_like, Poly) : Arguments to evaluate. Masked values and
                    np.nan will not be evaluated. If instance is Poly,
                    substitution on the variable is performed.
            kws (array_like, Poly) : Same as args, but the keys referred to the
                    variables names. If the number of dimensions are <=3, 'x',
                    'y' and 'z' respectably refer to the axes. Otherwise, the
                    keys are on the form 'x%d' where %d is an interger
                    representing the dimension.

        Returns:
            (ndarray, Poly) : If masked values are included in args, a Poly is
                    returned where the masked variables are retained.
                    Otherwise an array is returned.
        """
        if len(args)>self.dim:
            args = list(args[:self.dim])
        else:
            args = list(args) + [np.nan]*(self.dim-len(args))

        for key,val in kws.items():

            if key[0]=="q":
                index = int(key[1:])

                if index>=self.dim:
                    continue

            else:
                raise TypeError(
                    "Unexpeted keyword argument '%s'" % key)

            if args[index] not in (np.nan, np.ma.masked):
                raise TypeError(
                    "Multiple values for keyword argument '%s'" % index)

            args[index] = val

        return chaospy.poly.caller.call(self, args)


    def __contains__(self, y):
        """x.__contains__(y) <==> y in x"""

        if not isinstance(y, Poly):
            y = Poly(y)

        if self.shape==():
            if len(y.keys)>1:
                return NotImplemented
            return y.keys[0] in self.keys

        if len(y.shape)==len(self.shape)-1:
            return max(map(y.__eq__, self))

        if len(y.shape)<len(self.shape)-1:
            return max([y in s for s in self])

        return NotImplemented


    def __getitem__(self, I):
        shape = self.A[self.keys[0]][I].shape

        if isinstance(I, np.ndarray):
            I = I.tolist()

        if isinstance(I, (slice, int)):
            I = (I,)

        if not self.shape:
            subkey = I[0]
            subset = slice(None, None, None)

        elif len(self.shape)>=len(I):
            subkey = slice(None, None, None)
            subset = I

        elif len(self.shape)+1==len(I):
            subkey = I[-1]
            subset = I[:-1]

        else:
            subkey = slice(None, None, None)
            subset = I

        if isinstance(subkey, int):
            if not 0<=subkey<len(self.keys):
                raise IndexError("Index out of range")

            subkey = slice(subkey, subkey+1,None)


        A0 = {}
        for key in self.keys:
            tmp = self.A[key][subset]
            if not np.all(tmp==0):
                A0[key] = tmp

        A1 = {}
        for key in list(A0.keys())[subkey]:
            A1[key] = A0[key]

        return Poly(A1, self.dim, shape, self.dtype)

    def __iter__(self):
        """x.__iter__() <==> iter(x)"""
        A = self.A
        Pset = []

        if not self.shape:
            raise ValueError("not iterable")

        for i in range(self.shape[0]):

            out = {}
            for key in self.keys:

                if np.any(A[key][i]):
                    out[key] = A[key][i]

            Pset.append(Poly(out, self.dim, self.shape[1:],self.dtype))

        return Pset.__iter__()

    def __len__(self):
        """x.__len__() <==> len(x)"""

        if self.shape:
            return self.shape[0]
        return 1

    def __neg__(self):
        """x.__neg__() <==> -x"""

        A = self.A.copy()
        for key in self.keys:
            A[key] = -A[key]
        return Poly(A, self.dim, self.shape, self.dtype)

    def __pos__(self):
        return Poly(self.A, self.dim, self.shape, self.dtype)

    def __nonzero__(self):
        """x.__nonzero__() <==> x != 0"""

        if not self.A:
            return False
        return True


    def __pow__(self, n):
        """x.__pow__(y) <==> x**y"""
        if isinstance(n, (int, float, np.generic)):
            n = np.array(n)
            assert isinstance(n, np.ndarray)

        if isinstance(n, np.ndarray) and not n.shape:

            if abs(n-int(n))>1e-5:
                raise ValueError("Power of Poly must be interger")
            n = int(n)

            if n == 0:
                return Poly(
                    {(0,)*self.dim: np.ones(self.shape, dtype=int)},
                    self.dim, self.shape, None)

            self = self.copy()
            out = self
            for poly in range(1, n):
                out = out * self
            return out

        elif isinstance(n, (np.ndarray, list, tuple)):

            if not self.shape:
                out = [self.__pow__(n[i]) for i in range(len(n))]
                return Poly(out, self.dim, None, None)

            return Poly([self[i].__pow__(n[i])
                for i in range(len(n))], self.dim, None, None)

        raise ValueError("unknown type %s (%s)" % (str(n), type(n)))
        return NotImplemented



    def __str__(self):
        """x.__str__() <==> str(x)"""
        self = self.copy()

        # Array
        if self.shape:

            if len(self.shape) > 1:
                shape = self.shape
                self = chaospy.poly.shaping.flatten(self)
                P = np.reshape(np.array([str(p) for p in self],\
                    dtype=object), shape)
            else:
                P = np.array([str(p) for p in self], dtype=object)
            out = str(P.tolist())
            out = "".join(out.split("'"))
            return out

        if isinstance(VARNAME, str):
            basename = ["%s%d" % (VARNAME, d) for d in range(self.dim)]
        else:
            basename = list(VARNAME)

        out = []
        for key in self.keys[::-1]:

            o = ""
            coef = self.A[key]
            if np.sign(coef) == 1:
                o += "+"
            else:
                o += "-"
            if isinstance(coef, np.ndarray):
                o += str(abs(coef.item()))
            else:
                o += str(abs(coef))

            for j in range(self.dim):

                if key[j] == 1:
                    o += basename[j]
                if key[j] > 1:
                    o +="%s%s%d" % (basename[j], POWER, key[j])

            if not o:
                if self.dtype in (int, np.int16, np.int32, np.int64):
                    out = ["0"]
                else:
                    out = ["0.0"]
            elif o == "-0":
                o = "+0"
            if o == "-0.0":
                o = "+0.0"

            for b in basename:
                if o[4:5] == b[0] and o[1:].startswith("1.0"):
                    o = o[0] + o[4:]
                elif o[2:3] == b[0] and o[1:2] == "1":
                    o = o[0] + o[2:]
                elif ("/1%s" % b[0]) in o:
                    o = b.join(o.split("/1%s" % b))

            out.append(o)

        if not out:
            if self.dtype in (int, np.int16, np.int32, np.int64):
                out = ["0"]
            else:
                out = ["0.0"]
        if out[0][0]=="+": out[0] = out[0][1:]

        out = "".join(out)
        out = re.sub(r"(\d)(%s)" % re.escape(VARNAME), r"\1%s\2" % SEP, out)
        return out


    def copy(self):
        """Return a copy of the polynomial."""
        return Poly(self.A.copy(), self.dim, self.shape,
            self.dtype)

    def coeffs(self):
        out = np.array([self.A[key] for key in self.keys])
        out = np.rollaxis(out, -1)
        return out


def sort_key(val):
    """Sort key for sorting keys in grevlex order."""
    return np.sum((max(val)+1)**np.arange(len(val)-1, -1, -1)*val)


import chaospy as cp

if __name__=='__main__':
    import doctest
    doctest.testmod()
