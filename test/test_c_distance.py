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
        frames = int(config['Distance']['frames'])
        song_data = distance_matrix(fourier_folder=output_folder,
                                    multiprocess=False,
                                    frames=frames,
                                    distance_metric='positive')
        song_df = song_data.to_df()
        distance_folder = config['Folder']['Distance']
        frames = int(config['Distance']['frames'])
        if frames == 1:
            df = pd.read_csv(os.path.join(distance_folder, 'positive.csv'), sep=';', index_col=[0])
        else:
            df = pd.read_csv(os.path.join(distance_folder, 'positive.csv'), sep=';',
                             index_col=[0, 1]).set_index('song')
        pd.testing.assert_frame_equal(song_df, df)


if __name__ == '__main__':
    unittest.main()
