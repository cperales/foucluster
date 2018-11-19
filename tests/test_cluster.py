from foucluster.cluster import automatic_cluster, cluster_methods
import pandas as pd
import configparser
import os
from foucluster.distance import distance_dict
import numpy as np


def test_jump_method():
    """
    This tests use deterministic cluster methods
    and jump method to fix the optimal number
    of clusters.

    :return:
    """
    metrics = distance_dict.keys()
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Folder
    distance_folder = config['Folder']['Distance']
    cluster_folder = config['Folder']['Cluster']

    for metric in metrics:
        song_df = pd.read_csv(os.path.join(distance_folder,
                                           metric + '.csv'),
                              sep=';')
        song_df = song_df.set_index('Songs')
        for cluster_method in cluster_methods:
            cluster_df = automatic_cluster(dist_df=song_df.copy(deep=True),
                                           method=cluster_method)
            assert np.unique(cluster_df['Cluster'].values).shape[0] > 1


if __name__ == '__main__':
    test_jump_method()
