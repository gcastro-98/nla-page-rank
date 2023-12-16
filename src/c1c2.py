import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, diags
import time

from auxiliary import get_G, create_D

tol: float = 1e-15
m: float = 0.15


# ########################################################################
# PageRank FUNCTIONS
# ########################################################################

# STORING MATRICES

def PR_store() -> np.ndarray:
    """
    Compute the PR vector of M_m (storing matrices) using the power method,
    that is:
            $$ x_{k + 1} = (1 - m) G D x_k + e z^t x_k , $$
    until || x_{k + 1} - x_{k} || < tol.

    """
    # we initialize the necessary matrices
    G: coo_matrix = get_G()
    D: diags = create_D(G)

    A: coo_matrix = G.dot(D)
    n = A.shape[0]
    
    # and we define z knowing A.indices is the non-zero values' column positions
    e, z = np.ones(n), np.ones(n)/n
    z[np.unique(A.indices)] = m/n

    # and finally we start the power method algorithm for PR
    x_0 = np.zeros(n)
    x_k = np.ones(n) / n
    while np.linalg.norm(x_0 - x_k, np.inf) > tol:
        x_0 = x_k
        x_k = (1-m) * A.dot(x_0) + e * (np.dot(z, x_0))  # simply x = M_m * x

    return x_k / np.sum(x_k)


# WITHOUT STORING MATRICES

def PR_without_store() -> np.ndarray:
    """
    Computes the PR vector of M_m using the power method
    (without storing matrices). Leverages the Compressed Sparse Column (CSC)
    matrix structure and essentially:
    1. From the vectors that store the link matrix G obtain, for each
    j = 1, ... n; the set of indices L_j corresponding to pages having a link
    with page j.
    2. Compute the values n_j as the lenght of the set Lj .
    3. Iterate x_{k+1} = M_m x_{k} until || x_{k+1} - x_{k} || < tol using
    the code given in the statement (according to the developed idea).

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html

    """
    G: csc_matrix = csc_matrix(get_G())  # now G will be in CSC format
    n = G.shape[0]

    L, n_j = [], []
    for j in range(0, n):
        L_j = G.indices[G.indptr[j]: G.indptr[j+1]]  # webpages with link to j
        L.append(L_j)
        n_j.append(len(L_j))

    # and we start the power method algorithm for PR but without storing matrices
    x, xc = np.zeros(n), np.ones(n) / n
    while np.linalg.norm(x-xc, np.inf) > tol:
        # code from statement: starts
        xc = x
        x = np.zeros(n)
        for j in range(0, n):
            if n_j[j] == 0:
                x = x + xc[j] / n
            else:
                for i in L[j]:
                    x[i] = x[i] + xc[j] / n_j[j]
        # code from statement: ends

        x = (1 - m) * x + m / n

    return x / np.sum(x)


# ########################################################################
# MAIN FUNCTIONS
# ########################################################################

def _main_with_store() -> np.ndarray:
    """
    Compute the PR vector using the power method (storing matrices),
    report computation time and then, return the solution.
    """
    start = time.time()
    x = PR_store()
    print("With storing matrices:")
    print(4 * " " + f'Computation time: {time.time() - start} s')
    # print(4 * " " + f'PR vector: {x}')
    return x


def _main_without_store() -> np.ndarray:
    """
    Compute the PR vector of M_m, this time using the power method without
    storing matrices, report computation time and then, return the solution.
    """
    start = time.time()
    x = PR_without_store()
    print("Without storing matrices:")
    print(4 * " " + f'Computation time: {time.time() - start} s')
    # print(4 * " " + f'PR vector: {x}')
    return x


def main() -> None:
    x_store = _main_with_store()
    x_not_store = _main_without_store()
    print(f'Difference between solutions: {np.linalg.norm(x_store - x_not_store)}')


if __name__ == '__main__':
    main()
