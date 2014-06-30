auto_spectral_clustering
==========
Spectral clustering is one of the most popular modern clustering algorithms.
  Typically spectral clustering requires number of clusters manually.

auto_spectral_clustering predicts number of clusters based on the eigengap(often referred to as spectral gap).

References
==========
  A Tutorial on Spectral Clustering, 2007, Luxburg, Ulrike
  <http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf>

License
==========
New BSD License.

Dependencies
==========
auto_spectral_clustering is tested under Python 3.3.5.

It requires NumPy, SciPy, and scikit-learn.

Usage
==========
    from autosp import predict_k
    from sklearn.cluster import SpectralClustering

    k = predict_k(affinity_matrix)
    sc = SpectralClustering(n_clusters=k,
                            affinity="precomputed",
                            assign_labels="kmeans").fit(affinity_matrix)

    labels_pred = sc.labels_

Examples
==========
You can change number_of_blobs(artificial datasets) and test it!!

test.py
    
    if __name__ == "__main__":

        # Generate artificial datasets.
        number_of_blobs = 5  # You can change this!!
        data, labels_true = make_blobs(n_samples=number_of_blobs * 10,
                                       centers=number_of_blobs)

![Alt text](/fig/2.png)
![Alt text](/fig/7.png)
![Alt text](/fig/12.png)
![Alt text](/fig/20.png)