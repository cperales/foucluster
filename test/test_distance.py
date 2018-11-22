from foucluster.distance import distance_matrix
import unittest
import configparser
import warnings


class TestDistance(unittest.TestCase):

    def test_warp(self):
        """

        :return:
        """
        warnings.simplefilter("ignore")

        # Folder
        config = configparser.ConfigParser()
        config.read('config.ini')
        output_folder = config['Folder']['Output']
        song_df = distance_matrix(fourier_folder=output_folder,
                                  warp=100,
                                  upper_limit=6000.0,
                                  distance_metric='l2_norm')


if __name__ == '__main__':
    unittest.main()
