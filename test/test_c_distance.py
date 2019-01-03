from foucluster.distance import distance_matrix
import pandas as pd
import os
import unittest
import configparser
import warnings


class TestDistance(unittest.TestCase):

    def test_multiprocess(self):
        """
        """
        warnings.simplefilter("ignore")

        # Folder
        config = configparser.ConfigParser()
        config.read('config.ini')
        output_folder = config['Folder']['Output']
        song_df = distance_matrix(fourier_folder=output_folder,
                                  multiprocess=True,
                                  distance_metric='positive')
        distance_folder = config['Folder']['Distance']
        df = pd.read_csv(os.path.join(distance_folder, 'positive.csv'), sep=';')
        df = df.set_index('song')
        pd.testing.assert_frame_equal(song_df, df)


if __name__ == '__main__':
    unittest.main()
