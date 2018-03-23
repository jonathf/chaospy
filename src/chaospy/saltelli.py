"""
The Saltelli method

Code is built upon the code provided by Vinzenze Eck.
"""
import numpy


class Saltelli(object):
    """
    Buffer class to be able to retrieve Saltelli matrices.

    The core of the method relies on cross-combining the columns of two random
    matrices as part of a double expectation.

    Examples:
        >>> dist = chaospy.Iid(chaospy.Uniform(), 2)
        >>> generator = Saltelli(dist, 3, rule="H")

        >>> print(numpy.around(generator[(False, False)], 4))
        [[0.875  0.0625 0.5625]
         [0.5556 0.8889 0.037 ]]
        >>> print(numpy.around(generator[(False, True)], 4))
        [[0.875  0.0625 0.5625]
         [0.4444 0.7778 0.2222]]
        >>> print(numpy.around(generator[(True, False)], 4))
        [[0.125  0.625  0.375 ]
         [0.5556 0.8889 0.037 ]]
        >>> print(numpy.around(generator[(True, True)], 4))
        [[0.125  0.625  0.375 ]
         [0.4444 0.7778 0.2222]]
    """

    def __init__(self, dist, samples, poly=None, rule="R"):
        """
        Initialize the matrix generator.

        dist (chaopy.Dist):
            distribution to sample from.
        samples (int):
            The number of samples to draw for each matrix.
        poly (chaospy.Poly):
            If provided, evaluated samples through polynomials before returned.
        rule (str):
            Scheme for generating random samples.
        """
        self.dist = dist
        samples_ = dist.sample(2*samples, rule=rule)
        self.samples1 = samples_.T[:samples].T
        self.samples2 = samples_.T[samples:].T
        self.poly = poly
        self.buffer = {}

    def get_matrix(self, indices):
        """Retrieve Saltelli matrix."""
        new = numpy.empty(self.samples1.shape)
        for idx in range(len(indices)):
            if indices[idx]:
                new[idx] = self.samples1[idx]
            else:
                new[idx] = self.samples2[idx]

        if self.poly:
            new = self.poly(*new)
        return new

    def __getitem__(self, indices):
        """Shortcut to `get_matrix`."""
        assert len(self.dist) == len(indices)

        # uniquify:
        key = tuple(bool(idx) for idx in indices)

        if key in self.buffer:
            matrix = self.buffer[key]
        else:
            matrix = self.get_matrix(indices)
            self.buffer[key] = matrix

        return matrix


def Sens_m_sample(poly, dist, samples, rule="R"):
    """
    First order sensitivity indices estimated using Saltelli's method.

    Args:
        poly (chaospy.Poly):
            If provided, evaluated samples through polynomials before returned.
        dist (chaopy.Dist):
            distribution to sample from.
        samples (int):
            The number of samples to draw for each matrix.
        rule (str):
            Scheme for generating random samples.

    Return:
        (numpy.ndarray):
            array with `shape == (len(dist), len(poly))` where `sens[dim][pol]`
            is the first sensitivity index for distribution dimensions `dim` and
            polynomial index `pol`.

    Examples:
        >>> dist = chaospy.Iid(chaospy.Uniform(), 2)
        >>> poly = chaospy.basis(2, 2, dim=2)
        >>> print(poly)
        [q0^2, q0q1, q1^2]
        >>> print(numpy.around(Sens_m_sample(poly, dist, 10000, rule="M"), 4))
        [[0.008  0.0026 0.    ]
         [0.     0.6464 2.1321]]
    """
    dim = len(dist)

    generator = Saltelli(dist, samples, poly, rule=rule)

    zeros = [0]*dim
    ones = [1]*dim
    index = [0]*dim

    variance = numpy.var(generator[zeros], -1)

    matrix_0 = generator[zeros]
    matrix_1 = generator[ones]
    mean = .5*(numpy.mean(matrix_1) + numpy.mean(matrix_0))
    matrix_0 -= mean
    matrix_1 -= mean

    out = [
        numpy.mean(matrix_1*((generator[index]-mean)-matrix_0), -1) /
        numpy.where(variance, variance, 1)
        for index in numpy.eye(dim, dtype=bool)
    ]

    return numpy.array(out)


def Sens_m2_sample(poly, dist, samples, rule="R"):
    """
    Second order sensitivity indices estimated using Saltelli's method.

    Args:
        poly (chaospy.Poly):
            If provided, evaluated samples through polynomials before returned.
        dist (chaopy.Dist):
            distribution to sample from.
        samples (int):
            The number of samples to draw for each matrix.
        rule (str):
            Scheme for generating random samples.

    Return:
        (numpy.ndarray):
            array with `shape == (len(dist), len(dist), len(poly))` where
            `sens[dim1][dim2][pol]` is the correlating sensitivity between
            dimension `dim1` and `dim2` and polynomial index `pol`.

    Examples:
        >>> dist = chaospy.Iid(chaospy.Uniform(), 2)
        >>> poly = chaospy.basis(2, 2, dim=2)
        >>> print(poly)
        [q0^2, q0q1, q1^2]
        >>> print(numpy.around(Sens_m2_sample(poly, dist, 10000, rule="H"), 4))
        [[[ 0.008   0.0026  0.    ]
          [-0.0871  1.1516  1.2851]]
        <BLANKLINE>
         [[-0.0871  1.1516  1.2851]
          [ 0.      0.7981  1.38  ]]]
    """
    dim = len(dist)

    generator = Saltelli(dist, samples, poly, rule=rule)

    zeros = [0]*dim
    ones = [1]*dim
    index = [0]*dim

    variance = numpy.var(generator[zeros], -1)

    matrix_0 = generator[zeros]
    matrix_1 = generator[ones]
    mean = .5*(numpy.mean(matrix_1) + numpy.mean(matrix_0))

    matrix_0 -= mean
    matrix_1 -= mean

    out = numpy.empty((dim, dim)+poly.shape)
    for dim1 in range(dim):

        index[dim1] = 1
        matrix = generator[index]-mean
        out[dim1, dim1] = numpy.mean(
            matrix_1*(matrix-matrix_0),
            -1,
        ) / numpy.where(variance, variance, 1)

        for dim2 in range(dim1+1, dim):

            index[dim2] = 1

            matrix = generator[index]-mean

            out[dim1, dim2] = out[dim2, dim1] = numpy.mean(
                matrix_1*(matrix-matrix_0),
                -1,
            ) / numpy.where(variance, variance, 1)

            index[dim2] = 0

        index[dim1] = 0

    return out


def Sens_t_sample(poly, dist, samples, rule="R"):
    """
    Total order sensitivity indices estimated using Saltelli's method.

    Args:
        poly (chaospy.Poly):
            If provided, evaluated samples through polynomials before returned.
        dist (chaopy.Dist):
            distribution to sample from.
        samples (int):
            The number of samples to draw for each matrix.
        rule (str):
            Scheme for generating random samples.

    Return:
        (numpy.ndarray):
            array with `shape == (len(dist), len(poly))` where `sens[dim][pol]`
            is the total order sensitivity index for distribution dimensions
            `dim` and polynomial index `pol`.

    Examples:
        >>> dist = chaospy.Iid(chaospy.Uniform(0, 1), 2)
        >>> poly = chaospy.basis(2, 2, dim=2)
        >>> print(poly)
        [q0^2, q0q1, q1^2]
        >>> print(numpy.around(Sens_t_sample(poly, dist, 10000, rule="H"), 4))
        [[ 1.      0.2    -0.3807]
         [ 0.9916  0.9962  1.    ]]
    """
    generator = Saltelli(dist, samples, poly, rule=rule)

    dim = len(dist)
    zeros = [0]*dim
    variance = numpy.var(generator[zeros], -1)
    return numpy.array([
        1-numpy.mean((generator[~index]-generator[zeros])**2, -1,) /
        (2*numpy.where(variance, variance, 1))
        for index in numpy.eye(dim, dtype=bool)
    ])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
