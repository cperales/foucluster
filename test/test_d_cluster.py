from foucluster.cluster import automatic_cluster, cluster_methods, party_list
import pandas as pd
import configparser
import os
import numpy as np
import unittest
import warnings


class TestCluster(unittest.TestCase):

    def test_jump(self):
        """
        This tests use deterministic cluster methods
        and jump method to fix the optimal number
        of clusters.

        :return:
        """
        warnings.simplefilter("ignore")
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Folder
        distance_folder = config['Folder']['Distance']
        metric = 'l2_norm'
        song_df = pd.read_csv(os.path.join(distance_folder,
                                           metric + '.csv'),
                              sep=';')
        song_df = song_df.set_index('song')
        for cluster_method in cluster_methods:
            cluster_df = automatic_cluster(dist_df=song_df.copy(deep=True),
                                           method=cluster_method)
            self.assertGreater(np.unique(cluster_df['Cluster'].values).shape[0], 1)

    def test_party_list(self):
        """
        It should check that the order is different.

        :return:
        """
        warnings.simplefilter("ignore")
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Folder
        distance_folder = config['Folder']['Distance']

        # Metric (any metric)
        metric = 'l2_norm'
        song_df = pd.read_csv(os.path.join(distance_folder,
                                               metric + '.csv'),
                                  sep=';')
        song_df = song_df.set_index('song')
        first_song = song_df.index[0]
        song_list = party_list(song_df)
        order = True
        distance = 0.0
        for song in song_list.index:
            new_distance = song_df[first_song][song]
            if new_distance >= distance:
                distance = new_distance
            else:
                order = False
                break

        self.assertTrue(order)


if __name__ == "__main__":
    unittest.main()
