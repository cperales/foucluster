import unittest
from foucluster.plot import fourier_plot, \
    heatmap_song, song_plot, diff_plot
import configparser
import os
import json
import matplotlib.pyplot as plt


class TestPlot(unittest.TestCase):

    @staticmethod
    def _get_song(i=0):
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
        return j

    def test_diff(self):
        """

        :return:
        """
        song_1 = self._get_song(i=0)
        song_2 = self._get_song(i=1)
        diff_plot(song_1, song_2)
