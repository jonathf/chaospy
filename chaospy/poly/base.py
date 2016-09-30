import numpy as np

import chaospy as cp
from . import fraction as f
SUM = sum

__all__ = [
"Poly",
"call",
"decompose",
"dimsplit",
"dtyping",
"is_decomposed",
"setdim",
"substitute",
"setvarname",
]


# Name of variables
VARNAME = "q"
def setvarname(name):
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

    def __init__(self, A=None, dim=None, shape=None, dtype=None, V=0):
        """
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

        if V: print("\nConstruct poly out of:\n", A)

        if isinstance(A, (int, float)):
            A = np.array(A)

        if isinstance(A, Poly):

            dtype_ = A.dtype
            shape_ = A.shape
            dim_ = A.dim
            A = A.A.copy()

        elif isinstance(A, np.ndarray) and not A.shape:

            dtype_ = A.dtype
            shape_ = ()
            dim_ = 1
            A = {(0,):A}

        elif isinstance(A, dict):

            A = A.copy()

            if not A:
                dtype_ = int
                dim_ = 1
                shape_ = ()

            else:
                key = sorted(A.keys(), key=sort_key)[0]
                shape_ = np.array(A[key]).shape
                dim_ = len(key)
                dtype_ = dtyping(A[key])

        elif isinstance(A, f.frac):

            dtype_ = f.frac
            shape_ = A.shape
            dim_ = 1
            if isinstance(A.a, int):
                A = f.frac(np.array(A.a), np.array(A.b))
            A = {(0,): A}

        elif isinstance(A, (np.ndarray, list, tuple)):

            A = [Poly(a) for a in A]
            shape_ = (len(A),) + A[0].shape

            dtype_ = dtyping(*[_.dtype for _ in A])

            dims = np.array([a.dim for a in A])
            dim_ = np.max(dims)
            if dim_!=np.min(dims):
                A = [setdim(a, dim_) for a in A]

            d = {}
            for i in range(len(A)): # i over list of polys

                if V: print("Adding:", A[i], "(%d)" % i)
                for key in A[i].A: # key over exponents in each poly

                    if not key in d:
                        if V: print("creating key", key)
                        if dtype_==f.frac:
                            d[key] = f.frac(np.zeros(shape_))
                        else:
                            d[key] = np.zeros(shape_, dtype=dtype_)
                    d[key][i] = A[i].A[key]
                    if V: print("update", key, d[key])
            if V: print("creating master dict:\n", d)

            A = d

        else:
            raise TypeError(
                "Poly arg: 'A' is not a valid type " + repr(A))

        if dtype is None:
            dtype = dtype_

        if dtype == int:

            func1 = asint
            if shape is None:
                shape = shape_
            elif np.any(np.array(shape)!=shape_):
                ones = np.ones(shape, dtype=int)
                func1 = lambda x: asint(x*ones)

        elif dtype==f.frac:

            func1 = f.frac
            if shape is None:
                shape = shape_
            elif np.any(np.array(shape)!=shape_):
                ones = np.ones(shape, dtype=int)
                func1 = lambda x: f.frac(x*ones)

        else:

            func1 = lambda x:np.array(x, dtype=dtype)
            if shape is None:
                shape = shape_
            elif np.any(np.array(shape)!=shape_):
                ones = np.ones(shape, dtype=int)
                func1 = lambda x: 1.*x*ones

        func2 = lambda x:x
        if dim is None:
            dim = dim_
        elif dim<dim_:
            func2 = lambda x:x[:dim]
        elif dim>dim_:
            func2 = lambda x:x + (0,)*(dim-dim_)

        d = {}
        for key, val in A.items():
            d[func2(key)] = func1(val)
        A = d

        if isinstance(shape, int):
            shape = (shape,)

        # Remove empty elements
        for key in list(A.keys()):
            if np.all(A[key] == 0):
                del A[key]

        # assert non-empty container
        if not A:
            if dtype==float:
                dt = float
            else:
                dt = int
            A = {(0,)*dim: np.zeros(shape, dtype=dt)}

        self.keys = sorted(A.keys(), key=sort_key)
        self.dim = dim
        self.shape = shape
        self.dtype = dtype
        self.A = A

        if V: print("result", A)


    def __abs__(self):
        """x.__abs__() <==> abs(x)"""

        A = {key: abs(val) for key, val in self.A.items()}
        return Poly(A, self.dim, self.shape, self.dtype)

    def __add__(self, y):
        """x.__add__(y) <==> x+y"""

        if isinstance(y, Poly):

            if y.dim>self.dim:
                self = setdim(self, y.dim)
            elif y.dim<self.dim:
                y = setdim(y, self.dim)

            dtype = dtyping(self.dtype, y.dtype)

            d1 = self.A.copy()
            d2 = y.A.copy()

            if np.prod(y.shape)>np.prod(self.shape):
                shape = y.shape
                ones = np.ones(shape, dtype=int)
                for key in d1:
                    d1[key] = d1[key]*ones
            else:
                shape = self.shape
                ones = np.ones(shape, dtype=int)
                for key in d2:
                    d2[key] = d2[key]*ones

            if self.dtype==f.frac:
                for I in d2:
                    if d1.has_key(I):
                        d1[I] = d1[I] + d2[I]
                    else:
                        d1[I] = d2[I]
                out = d1
            else:
                for I in d1:
                    if I in d2:
                        d2[I] = d2[I] + d1[I]
                    else:
                        d2[I] = d1[I]
                out = d2

            return Poly(out, self.dim, shape, dtype)

        if isinstance(y, (float, list, tuple, int)):
            y = np.array(y)

        if isinstance(y, (np.ndarray, f.frac)):

            d = self.A.copy()
            dtype = dtyping(self.dtype, y.dtype)

            zero = (0,)*self.dim

            if zero not in d:
                d[zero] = np.zeros(self.shape, dtype=int)

            d[zero] = d[zero] + y

            if np.prod(y.shape) > np.prod(self.shape):

                ones = np.ones(y.shape, dtype=dtype)
                for key in d.keys():
                    d[key] = d[key]*ones

            return Poly(d, self.dim, None, dtype)

        return NotImplemented


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

        return call(self, args)


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

    def __div__(self, y):
        """x.__div__(y) <==> x/y"""
        print(repr(y))
        if isinstance(y, (float, int, f.frac)):
            return self.__mul__(y**-1)
        if isinstance(y, np.ndarray):
            y = np.asfarray(y)
            return self.__mul__(y**-1)
        print(123)
        return NotImplemented

    def __truediv__(self, y):
        return self.__div__(y)

    def __floordiv__(self, y):
        """x.__idiv__(y) <==> x//y"""
        if isinstance(y, f.frac):
            return self.__mul__(y**-1)
        if isinstance(y, (float, int, np.ndarray)):
            y = np.asfarray(np.array(y, dtype=int))
            return self.__mul__(y**-1)

        return NotImplemented

    def __eq__(self, y):
        """x.__eq__(y) <==> x==y"""
        return True - self.__ne__(y)

    def __ne__(self, y):

        if not isinstance(y, Poly):
            y = Poly(y)

        diff = abs(self - y)

        out = np.zeros(diff.shape, dtype=bool)

        for key in diff.keys:
            out = out + (diff.A[key]!=0)
        return out

    def __getitem__(self, I):
        """x.__getitem__(y) <==> x[y]"""

#          poly = list(self)
#          poly = np.array(poly)
#          return Poly(poly[I], self.dim, None, self.dtype)

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

        keys = sorted(A0.keys(), key=sort_key)
        dim = self.dim

        A1 = {}
        for key in keys[subkey]:
            A1[key] = A0[key]

        return Poly(A1, dim, shape, self.dtype)

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

            Pset.append(Poly(out, self.dim, self.shape[1:],
                self.dtype))

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


    def __mul__(self, y):
        """x.__mul__(y) <==> x*y"""
        if not isinstance(y, Poly):

            if isinstance(y, (float, int)):
                y = np.array(y)

            if not y.shape:
                A = self.A.copy()
                for key in self.keys:
                    A[key] = A[key]*y
                return Poly(A, self.dim, self.shape, self.dtype)

            y = Poly(y)

        if y.dim > self.dim:
            self = setdim(self, y.dim)

        elif y.dim<self.dim:
            y = setdim(y, self.dim)

        shape = [y,self][np.argmax([np.prod(y.shape),\
            np.prod(self.shape)])].shape

        dtype = dtyping(self.dtype, y.dtype)
        if self.dtype != y.dtype:

            if self.dtype==dtype:
                if dtype==f.frac:
                    y = asfrac(y)
                else:
                    y = asfloat(y)

            else:
                if dtype==f.frac:
                    self = asfrac(self)
                else:
                    self = asfloat(self)

        d = {}
        for I in y.A:
            for J in self.A:

                K = tuple(np.array(I)+np.array(J))
                d[K] = d.get(K,0) + y.A[I]*self.A[J]

        for K in list(d.keys()):
            if np.all(d[K]==0):
                del d[K]

        out = Poly(d, self.dim, shape, dtype)
        return out


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

    def __radd__(self, y):
        """x.__radd__(y) <==> y+x"""

        return self.__add__(y)


    def __rsub__(self, y):
        """x.__rsub__(y) <==> y-x"""

        return self.__neg__().__add__(y)


    def __rmul__(self, y):
        """x.__rmul__(y) <==> y*x"""

        return self.__mul__(y)


#      def __setitem__(self, I, y):
#          """x.__setitem__(i, y) <==> x[i]=y"""
#  
#          if not isinstance(y, Poly):
#              y = Poly(y)
#          if not y.shape:
#              y = Poly(y, y.dim, (1,), y.dtype)
#          if y.dim>self.dim:
#              self = setdim(self, y.dim)
#              A = self.A.copy()
#  
#          elif y.dim<self.dim:
#              y = setdim(y, self.dim)
#  
#  
#          if isinstance(I, (slice, int)):
#              I = (I,)
#  
#          if self.shape==(1,):
#              subkey = I[0]
#              subset = slice(None, None, None)
#  
#          elif len(self.shape)>=len(I):
#              subkey = slice(None, None, None)
#              subset = I
#  
#          elif len(self.shape)+1==len(I):
#              subkey = I[-1]
#              subset = I[:-1]
#  
#          else:
#              raise IndexError("Index out of range")
#  
#          if isinstance(subkey, int):
#              if not 0<=subkey<len(self.keys):
#                  raise IndexError("Index out of range")
#  
#              subkey = slice(subkey, subkey+1,None)
#  
#          A = self.A
#          for key in self.keys[:]:
#  
#              A[key][subset] = 0
#              if not np.any(A[key]):
#                  del A[key]
#                  self.keys.remove(key)
#  
#          shape = self.shape
#          for key in y.keys:
#  
#              if not A.has_key(key):
#                  A[key] = np.zeros(shape, dtype=int)
#                  self.keys.append(key)
#  
#              A[key][subset] = A[key][subset] + y.A[key][:]
#  


    def __str__(self):
        """x.__str__() <==> str(x)"""
        self = self.copy()

        # Array
        if self.shape:

            if len(self.shape)>1:
                shape = self.shape
                self = flatten(self)
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
            if isinstance(coef, f.frac):
                o += np.sign(coef.a)==1 and "+" or "-"
                o += str(abs(coef))
            else:
                o += np.sign(coef)==1 and "+" or "-"
                o += str(abs(coef.item()))

            for j in range(self.dim):

                if key[j]==1:
                    o += basename[j]
                if key[j]>1:
                    o +="%s^%d" % (basename[j], key[j])

            if not o or o=="-0": o = "+0"
            if o=="-0.0": o = "+0.0"

            for b in basename:
                if o[4:5]==b[0] and o[1:4]=="1.0": o = o[0] + o[4:]
                elif o[2:3]==b[0] and o[1:2]=="1": o = o[0] + o[2:]
                elif ("/1%s" % b[0]) in o: o = b.join(o.split("/1%s" % b))

            out.append(o)

        if not out: out = ["0"]
        if out[0][0]=="+": out[0] = out[0][1:]

        return "".join(out)


    def __sub__(self, y):
        """x.__sub__(y) <==> x-y"""

        if isinstance(y, Poly):
            return self.__add__(y.__neg__())
        return self.__add__(-y)


    def copy(self):
        """Return a copy of the array."""

        return Poly(self.A.copy(), self.dim, self.shape,
            self.dtype)

    def coeffs(self):
        out = np.array([self.A[key] for key in self.keys])
        out = np.rollaxis(out, -1)
        return out


def call(P, args):
    """
    Evaluate a polynomial along specified axes.

    Args:
        P (Poly) : Input polynomial.
        args (array_like, masked) : Argument to be evalutated.
                Masked values keeps the variable intact.

    Returns:
        (Poly, np.array) : If masked values are used the Poly is returned. Else
                an numpy array matching the polynomial's shape is returned.
    """
    args = list(args)

    # expand args to match dim
    if len(args)<P.dim:
        args = args + [np.nan]*(P.dim-len(args))
    elif len(args)>P.dim:
        raise ValueError("too many arguments")

    # Find and perform substitutions, if any
    x0,x1 = [],[]
    for i in range(len(args)):

        if isinstance(args[i], Poly):

            x0.append(Poly({tuple(np.eye(P.dim)[i]):np.array(1)}))
            x1.append(args[i])
            args[i] = np.nan
    if x0:
        P = call(P, args)
        return substitute(P, x0, x1)

    # Create masks
    masks = np.zeros(len(args), dtype=bool)
    for i in range(len(args)):
        if np.ma.is_masked(args[i]) \
            or np.any(args[i]!=args[i]):
            masks[i] = True
            args[i] = 0

    shape = np.array(args[np.argmax([np.prod(np.array(arg).shape)\
        for arg in args])]).shape
    args = np.array([np.ones(shape, dtype=int)*arg \
        for arg in args])

    A = {}
    for key in P.keys:

        key_ = np.array(key)*(1-masks)
        val = np.outer(P.A[key], np.prod((args.T**key_).T, \
                axis=0))
        val = np.reshape(val, P.shape + tuple(shape))
        val = np.where(val!=val, 0, val)

        mkey = tuple(np.array(key)*(masks))
        if not mkey in A:
            A[mkey] = val
        else:
            A[mkey] = A[mkey] + val
        if P.dtype==f.frac:
            A[mkey] = f.frac(A[mkey])

    out = Poly(A, P.dim, None, None)
    if out.keys and not np.sum(out.keys):
        out = out.A[out.keys[0]]
    elif not out.keys:
        out = np.zeros(out.shape, dtype=out.dtype)
    return out


def setdim(P, dim=None):
    """
    Adjust the dimensions of a polynomial.

    Output the results into Poly object

    Args:
        P (Poly) : Input polynomial
        dim (int) : The dimensions of the output polynomial. If omitted,
                increase polynomial with one dimension. If the new dim is
                smaller then P's dimensions, variables with cut components are
                all cut.

    Examples:
        >>> x,y = cp.variable(2)
        >>> P = x*x-x*y
        >>> print(cp.setdim(P, 1))
        q0^2
    """
    P = P.copy()

    ldim = P.dim
    if not dim:
        dim = ldim+1

    if dim==ldim:
        return P

    P.dim = dim
    if dim>ldim:

        key = np.zeros(dim, dtype=int)
        for lkey in P.keys:
            key[:ldim] = lkey
            P.A[tuple(key)] = P.A.pop(lkey)

    else:

        key = np.zeros(dim, dtype=int)
        for lkey in P.keys:
            if not SUM(lkey[ldim-1:]) or not SUM(lkey):
                P.A[lkey[:dim]] = P.A.pop(lkey)
            else:
                del P.A[lkey]

    P.keys = sorted(P.A.keys(), key=sort_key)
    return P


def decompose(P):
    """
    Decompose a polynomial to component form.

    In array missing values are padded with 0 to make decomposition compatible
    with `cp.sum(Q, 0)`.

    Args:
        P (Poly) : Input data.

    Returns:
        (Poly) : Decomposed polynomial with `P.shape==(M,)+Q.shape` where
                `M` is the number of components in `P`.

    Examples:
        >>> q = cp.variable()
        >>> P = cp.Poly([q**2-1, 2])
        >>> print(P)
        [q0^2-1, 2]
        >>> print(cp.decompose(P))
        [[-1, 2], [q0^2, 0]]
        >>> print(cp.sum(cp.decompose(P), 0))
        [q0^2-1, 2]
    """
    P = P.copy()

    if not P:
        return P

    out = [Poly({key:P.A[key]}) for key in P.keys]
    return Poly(out, None, None, None)


def is_decomposed(P):
    """
    Check if a polynomial (array) is on component form.

    Args:
        P (Poly) : Input data.

    Returns:
        (bool) : True if all polynomials in `P` are on component form.

    Examples:
        >>> x,y = cp.variable(2)
        >>> print(cp.is_decomposed(cp.Poly([1,x,x*y])))
        True
        >>> print(cp.is_decomposed(cp.Poly([x+1,x*y])))
        False
    """

    if P.shape:
        return min(map(is_decomposed, P))
    return len(P.keys)<=1


def dimsplit(P):
    """
    Segmentize a polynomial (on decomposed form) into it's dimensions.

    In array missing values are padded with 1 to make dimsplit compatible with
    `poly.prod(Q, 0)`.


    Args:
        P (Poly) : Input polynomial.

    Returns:
        (Poly) : Segmentet polynomial array where
                `Q.shape==P.shape+(P.dim+1,)`. The surplus element in `P.dim+1`
                is used for coeficients.

    Examples:
        >>> x,y = cp.variable(2)
        >>> P = cp.Poly([2, x*y, 2*x])
        >>> Q = cp.dimsplit(P)
        >>> print(Q)
        [[2, 1, 2], [1, q0, q0], [1, q1, 1]]
        >>> print(cp.prod(Q, 0))
        [2, q0q1, 2q0]
    """
    P = P.copy()

    if not is_decomposed(P):
        raise TypeError("Polynomial not on component form.")
    A = []

    dim = P.dim
    coef = P(*(1,)*dim)
    M = coef!=0
    zero = (0,)*dim
    ones = [1]*dim
    A = [{zero: coef}]

    if zero in P.A:

        del P.A[zero]
        P.keys.remove(zero)

    for key in P.keys:
        P.A[key] = (P.A[key]!=0)

    for i in range(dim):

        A.append({})
        ones[i] = np.nan
        Q = P(*ones)
        ones[i] = 1
        if isinstance(Q, np.ndarray):
            continue
        Q = Q.A

        if zero in Q:
            del Q[zero]

        for key in Q:

            val = Q[key]
            A[-1][key] = val

    A = [Poly(a, dim, None, P.dtype) for a in A]
    P = Poly(A, dim, None, P.dtype)
    P = P + 1*(P(*(1,)*dim)==0)*M

    return P

def substitute(P, x0, x1, V=0):
    """
    Substitute a variable in a polynomial array.

    Args:
        P (Poly) : Input data.
        x0 (Poly, int) : The variable to substitute. Indicated with either unit
                variable, e.g. `x`, `y`, `z`, etc. or through an integer
                matching the unit variables dimension, e.g. `x==0`, `y==1`,
                `z==2`, etc.
        x1 (Poly) : Simple polynomial to substitute `x0` in `P`. If `x1` is an
                polynomial array, an error will be raised.

    Returns:
        (Poly) : The resulting polynomial (array) where `x0` is replaced with
                `x1`.

    Examples:
        >>> x,y = cp.variable(2)
        >>> P = cp.Poly([y*y-1, y*x])
        >>> print(cp.substitute(P, y, x+1))
        [q0^2+2q0, q0^2+q0]

        With multiple substitutions:
        >>> print(cp.substitute(P, [x,y], [y,x]))
        [q0^2-1, q0q1]
    """
    if V: print("Replace", x0, "with", x1, "in", P)
    x0,x1 = map(Poly, [x0,x1])
    dim = np.max([p.dim for p in [P,x0,x1]])
    dtype = dtyping(P.dtype, x0.dtype, x1.dtype)
    P,x0,x1 = [setdim(p, dim) for p in [P,x0,x1]]

    if x0.shape:
        x0 = [x for x in x0]
    else:
        x0 = [x0]

    if x1.shape:
        x1 = [x for x in x1]
    else:
        x1 = [x1]

    # Check if substitution is needed.
    valid = False
    C = [x.keys[0].index(1) for x in x0]
    for key in P.keys:
        if np.any([key[c] for c in C]):
            valid = True
            break

    if not valid:
        return P

    dims = [tuple(np.array(x.keys[0])!=0).index(True) for x in x0]

    dec = is_decomposed(P)
    if not dec:
        P = decompose(P)

    P = dimsplit(P)

    shape = P.shape
    P = [p for p in flatten(P)]

    if V: print("Apriori:\n", P)

    for i in range(len(P)):
        for j in range(len(dims)):
            if P[i].keys and P[i].keys[0][dims[j]]:
                P[i] = x1[j].__pow__(P[i].keys[0][dims[j]])
                break

    if V: print("Aposteriori:\n", P)

    P = Poly(P, dim, None, dtype)
    P = reshape(P, shape)
    P = prod(P, 0)

    if not dec:
        P = sum(P, 0)

    return P



def dtyping(*args):
    """Find least common denominator dtype."""
    args = list(args)

    for i in range(len(args)):

        if isinstance(args[i], np.ndarray):
            args[i] = args[i].dtype
        elif isinstance(args[i],
            (float, f.frac, int)):
            args[i] = type(args[i])

    if Poly in args: return Poly

    if float in args: return float
    if np.dtype(float) in args: return float

    if object in args: return object
    if f.frac in args: return f.frac

    if int in args: return int
    if np.dtype(int) in args: return int

    if list in args: return list
    if tuple in args: return tuple

    raise ValueError("dtypes not recognised " + str(args))


# Collection compliant functions

def rollaxis(P, axis, start=0):

    A = P.A.copy()
    B = {}
    if P.dtype==f.frac:
        for key in P.keys:
            B[key] = f.rollaxis(A[key], axis, start)
    else:
        for key in P.keys:
            B[key] = np.rollaxis(A[key], axis, start)
    return Poly(B, P.dim, None, P.dtype)


def reshape(P, shape):

    A = P.A.copy()
    if P.dtype==f.frac:
        for key in P.keys:
            A[key] = f.reshape(A[key], shape)
        return Poly(A, P.dim, shape, f.frac)

    for key in P.keys:
        A[key] = np.reshape(A[key], shape)
    out = Poly(A, P.dim, shape, P.dtype)
    return out

def flatten(P):
    shape = int(np.prod(P.shape))
    return reshape(P, shape)

def sum(P, axis=None):

    if not P.A:
        if axis is None:
            return Poly({}, P.dim, (), P.dtype)
        shape = P.shape[:axis]+P.shape[axis+1:]
        return Poly({}, P.dim, shape, P.dtype)

    if isinstance(axis, int):

        l = len(P.shape)
        if axis<0: axis += l

        shape = [0]*(l-1)
        for i in range(l):
            if i<axis:
                shape[i] = P.shape[i]
            elif i>axis:
                shape[i-1] = P.shape[i]
        shape = tuple(shape)
    else:
        shape = ()

    A = P.A
    if P.dtype==f.frac:
        for key in P.keys:
            A[key] = f.sum(A[key], axis)
    else:
        for key in P.keys:
            A[key] = np.sum(A[key], axis)

    return Poly(A, P.dim, shape, P.dtype)

def prod(P, axis=None):

    if axis is None:
        P = flatten(P)
        axis = 0

    P = rollaxis(P, axis)
    Q = P[0]
    for p in P[1:]:
        Q = Q*p
    Q = Poly(Q, P.dim, None, P.dtype)
    return Q

def asfrac(P, limit=None):
    B = P.A.copy()
    for key in P.keys:
        B[key] = f.frac(B[key], 1, limit)

    out = Poly(B, P.dim, P.shape, f.frac)
    return out

def asint(P):

    if isinstance(P, f.frac):
        return P.a//P.b
    else:
        return np.array(P, dtype=int)

    B = P.A.copy()
    if P.dtype==f.frac:
        for key in P.keys:
            B[key] = B[key].a//B[key].b
    else:
        for key in P.keys:
            B[key] = np.array(B[key], dtype=int)

    out = Poly(B, P.dim, P.shape, int)
    return out



def toarray(P):
    shape = P.shape
    out = np.array([{} \
        for _ in range(np.prod(shape))], dtype=object)
    A = P.A.copy()
    for key in A.keys():

        A[key] = A[key].flatten()

        for i in range(np.prod(shape)):

            if not np.all(A[key][i]==0):
                out[i][key] = A[key][i]

    for i in range(np.prod(shape)):
        out[i] = Poly(out[i], P.dim, (), P.dtype)

    return out



def tolist(P):
    return toarray(P).tolist()

def mean(P, ax=None):

    A = P.A.copy()
    if P.dtype==f.frac:
        for key in P.keys:
            A[key] = f.mean(A[key], ax)
    else:
        for key in P.keys:
            A[key] = np.mean(A[key], ax)
    return Poly(A, P.dim, A[key].shape, P.dtype)

def var(P, ax=None):

    A = P.A.copy()
    if P.dtype==f.frac:
        for key in P.keys:
            A[key] = f.var(A[key], ax)
    else:
        for key in P.keys:
            A[key] = np.var(A[key], ax)
    return Poly(A, P.dim, A[key].shape, P.dtype)

def transpose(P):
    A = P.A.copy()
    if P.dtype==f.frac:
        for key in P.keys:
            A[key] = f.transpose(A[key])
    else:
        for key in P.keys:
            A[key] = np.transpose(A[key])
    return Poly(A, P.dim, P.shape[::-1], P.dtype)

def roll(P, shift, axis=None):
    A = P.A.copy()
    if P.dtype==f.frac:
        for key in P.keys:
            A[key] = f.roll(A[key], shift, axis)
    else:
        for key in P.keys:
            A[key] = np.roll(A[key], shift, axis)
    return Poly(A, P.dim, None, P.dtype)

def cumsum(P, axis=None):

    A = P.A.copy()
    if P.dtype==f.frac:
        for key in P.keys:
            A[key] = f.cumsum(A[key], axis)
    else:
        for key in P.keys:
            A[key] = np.cumsum(A[key], axis)
    return Poly(A, P.dim, P.shape, P.dtype)

def cumprod(P, axis=None):

    if np.prod(P.shape)==1:
        return P.copy()

    if axis is None:
        P = flatten(P)
        axis = 0

    Q = rollaxis(P, axis)
    Q = [_ for _ in Q]
    out, Q = Q[0], Q[1:]
    out.append(out[-1]*Q.pop(0))
    return Poly(Q, P.dim, P.shape, P.dtype)

def repeat(P, repeats, axis=None):

    A = P.A.copy()
    if P.dtype==f.frac:
        for key in P.keys:
            A[key] = f.repeat(A[key], repeats, axis)
    else:
        for key in P.keys:
            A[key] = np.repeat(A[key], repeats, axis)
    return Poly(A, P.dim, None, P.dtype)

def std(P, axis=None):
    A = P.A.copy()
    if P.dtype==f.frac:
        for key in P.keys:
            A[key] = f.std(A[key], axis)
    else:
        for key in P.keys:
            A[key] = np.std(A[key], axis)
    return Poly(A, P.dim, None, P.dtype)

def swapaxes(P, ax1, ax2):
    A = P.A.copy()
    if P.dtype==f.frac:
        for key in P.keys:
            A[key] = f.swapaxes(A[key], ax1, ax2)
    else:
        for key in P.keys:
            A[key] = np.swapaxes(A[key], ax1, ax2)
    return Poly(A, P.dim, None, P.dtype)

def trace(P, offset=0, ax1=0, ax2=1):
    A = P.A.copy()
    if P.dtype==f.frac:
        for key in P.keys:
            A[key] = f.trace(A[key], ax1, ax2)
    else:
        for key in P.keys:
            A[key] = np.trace(A[key], ax1, ax2)
    return Poly(A, P.dim, None, P.dtype)

def inner(*args):
    out = args[0]
    for arg in args[1:]:
        out = out * arg
    return sum(out)

def outer(*args):

    if len(args)>2:
        P1 = args[0]
        P2 = outer(*args[1:])
    elif len(args)==2:
        P1,P2 = args
    else:
        return args[0]

    if isinstance(P1, Poly) and isinstance(P2, Poly):

        if (1,) in (P1.shape, P2.shape):
            return P1*P2

        shape = P1.shape+P2.shape

        out = []
        for _ in flatten(P1):
            out.append(P2*_)

        out = reshape(Poly(out), shape)
        return out

    if isinstance(P1, (int, float, list, tuple)):
        P1 = np.array(P1)

    if isinstance(P2, (int, float, list, tuple)):
        P2 = np.array(P2)

    if isinstance(P1, Poly):
        A = P1.A
        B = {}
        for key in P1.keys:
            B[key] = outer(A[key], P2)
        shape = P1.shape+P2.shape
        dtype = dtyping(P1.dtype, P2.dtype)
        return Poly(B, P1.dim, shape, dtype)

    if isinstance(P2, Poly):
        A = P2.A
        B = {}
        for key in P2.keys:
            B[key] = outer(P1, A[key])
        shape = P1.shape+P2.shape
        dtype = dtyping(P1.dtype, P2.dtype)
        return Poly(B, P1.dim, shape, dtype)

def diag(P, k):

    A, B = P.A, {}
    for key in P.keys:
        B[key] = np.diag(A[key], k)

    return Poly(B, P.dim, None, P.dtype)


def asfloat(P):
    A = P.A.copy()
    for key in P.keys: A[key] = A[key]*1.
    return Poly(A, P.dim, P.shape, float)


def sort_key(val):
    """Sort key for sorting keys in grevlex order."""
    return np.sum(max(val)**np.arange(len(val)-1, -1, -1)*val)


if __name__=='__main__':
    import chaospy as cp
    import doctest
    doctest.testmod()
