# !/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.utils.graph import graph_laplacian
from sklearn.utils.arpack import eigsh
from sklearn.manifold.spectral_embedding_ import _set_diag


def predict_k(affinity_matrix):
    """
    Predict number of clusters based on the eigengap.

    Parameters
    ----------
    affinity_matrix : array-like or sparse matrix, shape: (n_samples, n_samples)
        adjacency matrix.
        Each element of this matrix contains a measure of similarity between two of the data points.

    Returns
    ----------
    k : integer
        estimated number of cluster.

    Note
    ---------
    If graph is not fully connected, zero component as single cluster.

    References
    ----------
    A Tutorial on Spectral Clustering, 2007
        Luxburg, Ulrike
        http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf

    """

    """
    If normed=True, L = D^(-1/2) * (D - A) * D^(-1/2) else L = D - A.
    normed=True is recommended.
    """
    normed_laplacian, dd = graph_laplacian(affinity_matrix, normed=True, return_diag=True)
    laplacian = _set_diag(normed_laplacian, 1)

    """
    n_components size is N - 1.
    Setting N - 1 may lead to slow execution time.
    A better approach is to set (N / 2)^(1/2)
    http://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set
    """
    n_components = affinity_matrix.shape[0] - 1

    """
    shift-invert mode
    The shift-invert mode provides more than just a fast way to obtain a few small eigenvalues.
    http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html

    The normalized Laplacian has eigenvalues between 0 and 2.
    I - L has eigenvalues between -1 and 1.
    """
    eigenvalues, eigenvectors = eigsh(-laplacian, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues = -eigenvalues[::-1]  # Reverse and sign inversion.

    max_gap = 0
    gap_pre_index = 0
    for i in range(1, eigenvalues.size):
        gap = eigenvalues[i] - eigenvalues[i - 1]
        if gap > max_gap:
            max_gap = gap
            gap_pre_index = i - 1

    k = gap_pre_index + 1

    return k