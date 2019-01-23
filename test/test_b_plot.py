import unittest
from foucluster.plot import song_plot, diff_plot
import configparser
import os
import json
from scipy.io.wavfile import read
import numpy as np


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
        name = list(j.keys())[0]
        song = j[name]
        return song, name

    @staticmethod
    def _get_song(i=0):
        """

        :return:
        """
        config = configparser.ConfigParser()
        config.read('config.ini')
        song_folder = config['Folder']['Temp']
        first_song = os.listdir(song_folder)[i]
        rate, aud_data = read(os.path.join(song_folder,
                                           first_song))
        # Should be mono
        if len(aud_data) != len(aud_data.ravel()):
            aud_data = np.mean(aud_data, axis=1)
        return aud_data,first_song

    def test_diff(self):
        """

        :return:
        """
        config = configparser.ConfigParser()
        config.read('config.ini')
        image_folder = config['Folder']['Image']
        song_1, name_1 = self._get_series(i=0)
        song_2, name_2 = self._get_series(i=1)
        diff_plot(song_1, song_2,
                  filename=name_1.split()[2].split('.')[0] + name_2.split()[2].split('.')[0],
                  folder=image_folder)

    def test_song(self):
        """

        :return:
        """
        config = configparser.ConfigParser()
        config.read('config.ini')
        image_folder = config['Folder']['Image']
        aud_data, name = self._get_song()
        song_plot(aud_data,
                  filename=name.split('.')[0],
                  folder=image_folder)


if __name__ == '__main__':
    unittest.main()
