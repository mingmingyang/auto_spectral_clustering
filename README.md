auto_spectral_clustering
==========
Spectral clustering is one of the most popular modern clustering algorithms.
  Typically spectral clustering requires number of clusters manually.

auto_spectral_clustering predict number of clusters based on the eigengap(often referred to as spectral gap).

References
==========
  A Tutorial on Spectral Clustering, 2007, Luxburg, Ulrike
  <http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf>

Dependencies
==========
auto_spectral_clustering is tested under Python 3.3.5.

It requires NumPy, SciPy, and scikit-learn.

Examples
==========
You can change number_of_clusters(artificial datasets) and test it!!

test.py
    
    if __name__ == "__main__":

        # Generate artificial datasets.
        number_of_clusters = 5  # You can change this!!
        data, labels_true = make_blobs(n_samples=number_of_clusters * 10,
                                    centers=number_of_clusters)

![Alt text](/fig/2.png)
![Alt text](/fig/7.png)
![Alt text](/fig/12.png)
![Alt text](/fig/20.png)