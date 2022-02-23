import numpy as np


def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    return (np.matmul(X, Y.T) + c)**p


def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::

            K(x, y) = exp(-gamma ||x-y||^2)

        for each pair of rows x in X and y in Y.

        ||x-y||^2 = X.T * X + Y.T * Y - 2X.T * Y

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    norms_x = (X ** 2).sum(axis=1)
    norms_y = (Y ** 2).sum(axis=1)
    dists_sq = np.abs(norms_x.reshape(-1, 1) + norms_y - 2 * np.matmul(X, Y.T))
    return np.exp(-gamma * dists_sq)
