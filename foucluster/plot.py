import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def heatmap_song(song_df,
                 image_name,
                 image_folder=None):
    """
    Plot heatmap of a distance dataframe.

    :param song_df:
    :param image_name:
    :param image_folder:
    :return:
    """
    fig, ax = plt.subplots()
    sns.heatmap(song_df)
    fig.subplots_adjust(left=0.35,
                        bottom=0.35,
                        right=1.0,
                        top=0.95)
    plt.savefig(os.path.join(image_folder, image_name) + '.png')
    plt.close()


def fourier_plot(freq, features,
                 folder=None,
                 filename=None):
    """
    
    """
    fig = plt.figure(1)
    # Turn interactive plotting off
    plt.ioff()
    plt.plot(freq, features)
    plt.xlabel('frequency')
    plt.ylabel('amplitude')
    if filename is not None:
        f = '' if folder is None else folder
        plt.savefig(os.path.join(f,
                                 filename + '.png'))
    plt.close(fig)


def song_plot(aud_data,
              folder=None,
              filename=None):
    fig = plt.figure(1)
    # Turn interactive plotting off
    plt.ioff()
    plt.plot(aud_data)
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.xticks([])
    plt.yticks([])
    if filename is not None:
        f = '' if folder is None else folder
        plt.savefig(os.path.join(f,
                                 filename + '.png'))
    plt.close(fig)


def diff_plot(song_1, song_2, filename=None, folder=None):
    """
    Plot the difference between two series.

    :param dict song_1:
    :param dict song_2:
    :param str filename:
    :param str folder:
    :return:
    """
    x_1 = list(song_1.keys())
    y_1 = list(song_1.values())
    x_2 = list(song_2.keys())
    y_2 = list(song_2.values())
    y_2_interp = np.interp(x_1,
                           x_2,
                           y_2)
    fig, ax_1 = plt.subplots()
    ax_1.fill_between(x_1, y_1, y_2_interp)
    if filename is not None:
        f = '' if folder is None else folder
        plt.savefig(os.path.join(f,
                                 filename + '.png'))
    plt.close(fig)
