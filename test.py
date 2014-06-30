# !/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph

from autosp import predict_k


def main(affinity_matrix):
    """Test autosp

    Parameters
    ----------
    affinity_matrix : array-like or sparse matrix, shape: (n_samples, n_samples)
        adjacency matrix.
        Each element of this matrix contains a measure of similarity between two of the data points.

    Returns
    ----------
    labels: array of integers, shape: n_samples
        The labels of the clusters.
    k : integer
        estimated number of cluster.
    """

    k = predict_k(affinity_matrix)

    sc = SpectralClustering(n_clusters=k,
                            affinity="precomputed",
                            assign_labels="kmeans").fit(affinity_matrix)

    labels = sc.labels_

    return labels, k


def color_map(labels):
    """Achieve "some" consistency of color between true labels and pred labels.


    Parameters
    ----------
    labels : array of integers, shape: n_samples
        The labels of the clusters.

    Returns
    ----------
    color_map : dict object {integer: integer}
        The map of labels.
    """

    color_map = {}

    i = 0
    v = 0
    while v != max(labels) + 1:
        if labels[i] in color_map:
            pass
        else:
            color_map[labels[i]] = v
            v += 1
        i += 1

    return color_map

if __name__ == "__main__":

    # Generate artificial datasets.
    number_of_blobs = 20  # You can change this!!
    data, labels_true = make_blobs(n_samples=number_of_blobs * 10,
                                   centers=number_of_blobs)

    # Calculate affinity_matrix.
    connectivity = kneighbors_graph(data, n_neighbors=10)
    affinity_matrix = 0.5 * (connectivity + connectivity.T)

    labels_pred, k = main(affinity_matrix)

    print("%d blobs(artificial datasets)." % number_of_blobs)
    print("%d clusters(predicted)." % k)

    # Plot.
    from pylab import *
    t_map = color_map(labels_true)
    t = [t_map[v] for v in labels_true]

    p_map = color_map(labels_pred)
    p = [p_map[v] for v in labels_pred]


    subplot(211)
    title("%d blobs(artificial datasets)." % number_of_blobs)
    scatter(data[:, 0], data[:, 1], s=150, c=t)

    subplot(212)
    title("%d clusters(predicted)." % k)
    scatter(data[:, 0], data[:, 1], s=150, c=p)

    show()
