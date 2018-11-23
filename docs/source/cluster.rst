Cluster of songs using distances
=========================================================

Once the distance metric is calculated (the output from
`foucluster.distance.distance_matrix`), this distances between
the songs are used as features for applying clustering.

Several methodologies from `sklearn` are imported:

- Indicating the number of clusters (`KMeans`,
    `AgglomerativeClustering`, `SpectralClustering`).
- Without the number of clusters (`AffinityPropagation`,
    `MeanShift`).

For the first type of clusters,

.. autofunction:: foucluster.cluster.determinist_cluster

For both types of clusters,

.. autofunction:: foucluster.cluster.automatic_cluster

When an algorithm which needs the number of clusters, like `KMeans`,
is used with `automatic_cluster`, it calls to
`jump method <https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set#An_information%E2%80%93theoretic_approach>`_ to calculate the number
of clusters.

.. autofunction:: foucluster.cluster.jump_method
