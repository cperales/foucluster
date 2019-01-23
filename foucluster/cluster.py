from sklearn import cluster
from sklearn.preprocessing import minmax_scale, scale
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np
from itertools import groupby

eps = 10**(-10)

n_cluster_methods = {'AgglomerativeClustering': cluster.AgglomerativeClustering,
                     'SpectralClustering': cluster.SpectralClustering,
                     'KMeans': cluster.KMeans}

non_n_cluster_methods = {'AffinityPropagation': cluster.AffinityPropagation,
                         'MeanShift': cluster.MeanShift}


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
    if not isinstance(dist_df, pd.DataFrame):
        dist_df = dist_df.to_df().T
    dist_values = dist_df.values
    df_matrix = scale(dist_values)
    y = n_cluster_methods[method](n_clusters=n_clusters).fit_predict(df_matrix)
    cluster_series = pd.Series(y, index=dist_df.index)
    return cluster_series


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
    if not isinstance(dist_df, pd.DataFrame):
        dist_df = dist_df.to_df().T
    dist_values = dist_df.values
    df_matrix = scale(dist_values)
    if method in n_cluster_methods.keys():
        n_clusters = jump_method(dist_df=df_matrix)
        clf = n_cluster_methods[method](n_clusters=n_clusters)
    else:
        clf = non_n_cluster_methods[method]()
    y = clf.fit_predict(df_matrix)
    cluster_series = pd.Series(y, index=dist_df.index)
    return cluster_series


def jump_method(dist_df, n_max=50):
    """
    Method based on information theory to determine best
    number of clusters.

    :param np.array dist_df:
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

    # Avoiding let an instance alone
    instance_alone = True
    while instance_alone is True:
        y = cluster.KMeans(n_clusters=n_cluster).fit_predict(dist_df)
        group_member = [len(list(group)) for key, group in groupby(np.sort(y))]
        if np.min(group_member) > 1 or n_cluster == 2:
            instance_alone = False
        else:
            n_cluster -= 1

    return n_cluster


def score_cluster(cluster_df):
    """
    When `automatic_cluster` is used, then the clusters must be
    grouped into the categories we want into predict, in order to score
    our method.

    :param pandas.DataFrame cluster_df:
    :return: accuracy score. cluster_df have now `Cluster_corrected` column.
    """
    accurate_class = [int(n[0][0]) for n in cluster_df.index.tolist()]
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


def party_list(song_df, song=None):
    """
    A list of song of all the songs from the cluster dataframe
    sorted, from similarity between them.

    :param pandas.DataFrame song_df:
    :param str song:
    :return:
    """
    song_df_rev = song_df.T
    if song is None or song not in song_df_rev.index:
        song = song_df_rev.index[0]
    # TODO: to implement
    final_index = list(song_df_rev.sort_values(song, axis='columns')[song].index)
    return final_index


def zero_scale(X):
    """

    :param numpy.array X:
    :return:
    """
    n, m = X.shape
    x = np.empty_like(X)
    for j in range(m):
        feature_column = X[:, j]
        max_value = feature_column.max()
        min_value = feature_column.min()
        feature_column = (feature_column - min_value) / (max_value - min_value)
        x[:, j] = feature_column
    return x
