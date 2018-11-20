from sklearn import cluster
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np

eps = 10**(-10)

n_cluster_methods = {'cluster.AgglomerativeClustering': cluster.AgglomerativeClustering,
                     'cluster.SpectralClustering': cluster.SpectralClustering,
                     'cluster.KMeans': cluster.KMeans}

non_n_cluster_methods = {'cluster.AffinityPropagation': cluster.AffinityPropagation,
                         'cluster.MeanShift': cluster.MeanShift}


cluster_methods = n_cluster_methods.copy()
cluster_methods.update(non_n_cluster_methods)


def determinist_cluster(dist_df, method, n_clusters):
    """
    Clustering of the songs from the dataframe, indicating
    the number of clusters to use.

    :param pandas.DataFrame dist_df:
    :param str method: name of the sklearn.cluster.

            - cluster.AgglomerativeClustering.
            - cluster.SpectralClustering.
            - cluster.KMeans.

    :param int n_clusters:
    :return: pandas.DataFrame with a column with clusters.
    """
    df_matrix = minmax_scale(dist_df)
    y = n_cluster_methods[method](n_clusters=n_clusters).fit_predict(df_matrix)
    cluster_df = dist_df.copy(deep=True)
    cluster_df['Cluster'] = pd.Series(y, index=cluster_df.index)
    return cluster_df


def automatic_cluster(dist_df, method):
    """

    :param pd.DataFrame dist_df:
    :param str method: name of the sklearn.cluster.

            - cluster.AffinityPropagation.
            - cluster.MeanShift.
            - cluster.AgglomerativeClustering.
            - cluster.SpectralClustering.
            - cluster.KMeans.

    :return: pandas.DataFrame with a column with clusters.
    """
    df_matrix = minmax_scale(dist_df)
    if method in n_cluster_methods.keys():
        n_clusters = jump_method(dist_df=df_matrix)
        clf = n_cluster_methods[method](n_clusters=n_clusters)
    else:
        clf = non_n_cluster_methods[method]()
    y = clf.fit_predict(df_matrix)
    cluster_df = dist_df.copy(deep=True)
    cluster_df['Cluster'] = pd.Series(y, index=cluster_df.index)
    return cluster_df


def jump_method(dist_df, n_max=50):
    """
    Method based on information theory to determine best
    number of clusters.

    :param pandas.DataFrame dist_df:
    :param int n_max: maximum number of clusters to test.
    :return: optimal number of clusters
    """
    dim = dist_df.shape[0]
    if n_max > dim:
        n_max = dim
    Y = dim / 2
    distortions = np.empty(n_max + 1)
    jump_vector = np.empty(n_max)
    distortions[0] = 0.0
    for k in range(1, n_max + 1):
        kmean_model = cluster.KMeans(n_clusters=k).fit(dist_df)
        distortion = np.min(cdist(dist_df,
                                  kmean_model.cluster_centers_,
                                  'euclidean').ravel()) / dim + eps
        distortions[k] = distortion**(- Y)
        jump_vector[k - 1] = distortions[k] - distortions[k - 1]
    n_cluster = np.argmax(jump_vector) + 1
    return n_cluster


def score_cluster(cluster_df):
    """
    When `automatic_cluster` is used, then the clusters must be
    grouped into the categories we want into predict, in order to score
    our method.

    :param pandas.DataFrame cluster_df:
    :return: accuracy score. cluster_df have now `Cluster_corrected` column.
    """
    accurate_class = [int(n[0]) for n in cluster_df.index.tolist()]
    accurate_class -= np.unique(accurate_class)[0]
    # Move to 0, 1, ... notation
    accurate_class = np.array(accurate_class, dtype=int)
    cluster_class = np.array(cluster_df['Cluster'].tolist(), dtype=int)
    # Find correspondences between given classes and cluster classes
    correspondence_dict = {}

    for p in np.unique(cluster_class):
        max_c = 0.0
        pos_p = cluster_class == p
        for e in np.unique(accurate_class):
            pos_e = accurate_class == e
            c = (pos_p == pos_e).sum()
            if c > max_c:
                correspondence_dict.update({p: e})
                max_c = c
    # Finding the accuracy
    cluster_class_corrected = [correspondence_dict[p] for p in cluster_class]
    cluster_df['Cluster_corrected'] = pd.Series(cluster_class_corrected,
                                                index=cluster_df.index)
    score_vector = [e == p_c for e, p_c in
                    zip(accurate_class, cluster_class_corrected)]
    return np.average(score_vector)


def party_list(cluster_df, song=None):
    """
    A list of song of all the songs from the cluster dataframe
    sorted, from similarity between them.

    :param pandas.DataFrame cluster_df:
    :param str song:
    :return:
    """
    if song is None or song not in cluster_df.columns:
        song = cluster_df.index[0]
    print(cluster_df.sort_values(song)[song])
