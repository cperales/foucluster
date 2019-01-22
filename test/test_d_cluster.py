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
                              sep=';',
                              index_col=[0, 1])
        for cluster_method in cluster_methods:
            cluster_df = automatic_cluster(dist_df=song_df.copy(deep=True),
                                           method=cluster_method)
            self.assertGreater(np.unique(cluster_df['Cluster'].values).shape[0], 1)

    def todo_party_list(self):
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
                              sep=';',
                              index_col=[0, 1])
        first_song = song_df.columns[0]
        song_list, song_df_rev = party_list(song_df)
        order = True
        distance = 0.0
        for song in song_list:
            new_distance = np.mean(song_df[first_song][song])
            if new_distance >= distance:
                distance = new_distance
            else:
                order = False
                break

        self.assertTrue(order)


if __name__ == "__main__":
    unittest.main()
