from foucluster.cluster import automatic_cluster, cluster_methods, party_list
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

    for metric in metrics:
        song_df = pd.read_csv(os.path.join(distance_folder,
                                           metric + '.csv'),
                              sep=';')
        song_df = song_df.set_index('Songs')
        for cluster_method in cluster_methods:
            cluster_df = automatic_cluster(dist_df=song_df.copy(deep=True),
                                           method=cluster_method)
            assert np.unique(cluster_df['Cluster'].values).shape[0] > 1


def test_party_list():
    """
    It should check that the order is different.

    :return:
    """
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Folder
    distance_folder = config['Folder']['Distance']

    # Metric (any metric)
    metric = 'l2_norm'
    song_df = pd.read_csv(os.path.join(distance_folder,
                                           metric + '.csv'),
                              sep=';')
    song_df = song_df.set_index('Songs')
    first_song = song_df.index[0]
    song_list = party_list(song_df, song=first_song)
    order = True
    distance = 0.0
    for song in song_list.index:
        new_distance = song_df[first_song][song]
        if new_distance >= distance:
            distance = new_distance
        else:
            order = False
            break

    assert order is True


if __name__ == '__main__':
    test_jump_method()
    test_party_list()
