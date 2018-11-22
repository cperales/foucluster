import unittest
from foucluster.plot import fourier_plot, \
    heatmap_song, song_plot, diff_plot
import configparser
import os
import json
import matplotlib.pyplot as plt
from scipy.io.wavfile import read


class TestPlot(unittest.TestCase):

    @staticmethod
    def _get_series(i=0):
        """

        :return:
        """
        config = configparser.ConfigParser()
        config.read('config.ini')
        fourier_folder = config['Folder']['Output']
        first_file = os.path.join(fourier_folder,
                                  os.listdir(fourier_folder)[i])
        with open(first_file, 'r') as b:
            j = json.load(b)
        song = j[list(j.keys())[0]]
        return song

    @staticmethod
    def _get_song(i=0):
        """

        :return:
        """
        config = configparser.ConfigParser()
        config.read('config.ini')
        song_folder = config['Folder']['Output']
        first_song = os.path.join(song_folder,
                                  os.listdir(song_folder)[i])
        rate, aud_data = read(os.path.join(first_song,
                                           song_folder))
        # Should be mono
        if len(aud_data) != len(aud_data.ravel()):
            aud_data = np.mean(aud_data, axis=1)
        return aud_data

    def test_diff(self):
        """

        :return:
        """
        song_1 = self._get_series(i=0)
        song_2 = self._get_series(i=1)
        diff_plot(song_1, song_2)

    def test_song(self):
        """

        :return:
        """


if __name__ == '__main__':
    unittest.main()
