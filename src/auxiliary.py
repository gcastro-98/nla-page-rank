import numpy as np
from scipy.io import mmread
from scipy.sparse import coo_matrix, diags


# ########################################################################
# AUXILIARY FUNCTIONS
# ########################################################################

def get_G(link_matrix: str = 'p2p-Gnutella30.mtx') -> coo_matrix:
    """
    Load the locally serialized link matrix G=(g_{ij}) with g_{ij} = 0 or 1
    if there is link between i and j. Return it as sparse COOrdinate matrix.

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
    """
    return mmread(link_matrix)


def create_D(G: coo_matrix) -> diags:
    """
    Create D, from the G graph matrix, in form of diagonal sparse matrix and
    using the out-degree values n_j

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.diags.html
    """
    n_j = np.sum(G, axis=0)  # the out-degree is simply the row sum
    # and we then obtain the diagonal D matrix
    d = np.zeros(G.shape[0])
    for i in range(G.shape[0]):
        if n_j[0, i] == 0:
            d[i] = 0
        else:
            d[i] = 1 / n_j[0, i]
    return diags(d)  # returning it as `diags` diagonal sparse matrix
