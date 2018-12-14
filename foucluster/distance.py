import json
import glob
import os

import numpy as np
import pandas as pd
from copy import deepcopy
from .transform import limit_by_freq, dict_to_array

# sqrt(2) with default precision np.float64
_SQRT2 = np.sqrt(2)

# DUMMY VECTOR CLASS
class Data:
    """
    Dummy class in order to store into the dataframe.
    """
    def __init__(self, vector):
        self.vector = vector
        self.abs_distance = np.sum(self.vector)

    def unpack(self):
        return self.abs_distance



# DISTANCE METRICS

def positive_error(x, y):
    """
    :param np.array x:
    :param np.array y:
    :return:
    """
    return np.sum(np.abs(x - y))


def hellinger(x, y):
    """
    :param np.array x:
    :param np.array y:
    :return:
    """
    return np.linalg.norm(np.sqrt(x) / np.sum(x) -
                          np.sqrt(y) / np.sum(y)) / _SQRT2


def l2_norm(x, y):
    """
    L2 norm, adapted to dtw format
    :param x:
    :param y:
    :return: euclidean norm
    """
    return np.linalg.norm(x - y)


def integrate(x, y):
    """
    :param x:
    :param y:
    :return:
    """
    diff = np.abs(x - y)
    return np.trapz(diff)


distance_dict = {'positive': positive_error,
                 'hellinger': hellinger,
                 'l2_norm': l2_norm,
                 'integrate': integrate}


def warp_distance(distance_metric, x, y, warp=200):
    """

    :param str distance_metric:
    :param np.array x:
    :param np.array y:
    :param int warp:
    :return:
    """
    # Selecting the array
    distance_func = distance_dict[distance_metric]
    # Copying the value
    x_copy = deepcopy(x)
    y_copy = deepcopy(y)
    # Starting the warping
    min_diff = distance_func(x, y)
    for i in range(1, int(warp)):
        # Moving forward
        forward_diff = distance_func(x_copy[i:], y_copy[:-i])
        if forward_diff < min_diff:
            min_diff = forward_diff
        # Moving backward
        backward_diff = distance_func(x_copy[:-i], y_copy[i:])
        if backward_diff < forward_diff:
            min_diff = backward_diff
    return min_diff


def pair_distance(freq_x,
                  features_x,
                  freq_y,
                  features_y,
                  frames=None,
                  distance_metric='l2_norm'):
    """
    Distance between song x (with frequencies and features)
    and song y is calculated.

    :param numpy.array freq_x: frequencies of the song x.
    :param numpy.array features_x: features (fourier amplitude) of song x.
    :param numpy.array freq_y: frequencies of the song y.
    :param numpy.array features_y: features (fourier amplitude) of song y.
    :param frames: number of frames to calculate distances. If None,
        only one frame is considered
    :param str distance_metric: name of the metric to use. Options are:

            - 'positive': positive_error.
            - 'hellinger': hellinger.
            - 'l2_norm': l2_norm.
            - 'integrate': integrate.

    :return: distance in float.
    """

    if frames is None:
        frames = 1

    freq_x_frames = np.array_split(freq_x, frames)
    features_x_frames = np.array_split(features_x, frames)

    distance_array = np.empty(frames)
    for frame in range(frames):
        # Get the frames
        freq_x_frame = freq_x_frames[frame]
        features_x_frame = features_x_frames[frame]
        # Interpolate to get features from song y
        features_y_frame = np.interp(freq_x_frame,
                                     freq_y,
                                     features_y)
        distance = distance_dict[distance_metric](features_x_frame,
                                                  features_y_frame)
        distance_array[frame] = distance  # / np.max(features_x_frame)

    return distance_array


def distance_matrix(fourier_folder,
                    frames=None,
                    upper_limit=6000.0,
                    distance_metric='l2_norm'):
    """
    A distance matrix with all the songs of a folder
    can be calculated.

    :param fourier_folder:
    :param frames:
    :param upper_limit:
    :param distance_metric:
    :return:
    """
    merged_file = os.path.join(fourier_folder, 'merged_file.json')
    if os.path.isfile(merged_file):
        os.remove(merged_file)
    read_files = glob.glob(os.path.join(fourier_folder, '*.json'))
    merged_file_list = [json.load(open(f)) for f in read_files]
    merged_file = merged_file_list[0]
    [merged_file.update(d) for d in merged_file_list]

    # Creating a squared DataFrame as matrix distance
    song_names = list(merged_file.keys())
    df = pd.DataFrame(columns=song_names + ['Songs'])
    df['Songs'] = song_names
    df = df.set_index('Songs')
    for i in range(len(song_names)):
        song_x = song_names[i]
        freq_x, features_x = dict_to_array(merged_file[song_x])
        # Filtering frequencies
        freq_x, features_x = limit_by_freq(freq_x,
                                           features_x,
                                           upper_limit=upper_limit)
        for j in range(i, len(song_names)):
            song_y = song_names[j]
            if j > i:
                freq_y, features_y = dict_to_array(merged_file[song_y])
                distance = pair_distance(freq_x=freq_x,
                                         features_x=features_x,
                                         freq_y=freq_y,
                                         features_y=features_y,
                                         frames=frames,
                                         distance_metric=distance_metric)
                # Save the symmetric option

            else:  # j == i:
                distance = np.zeros(frames)

            vector_to_save = Data(distance)
            df.loc[song_x, song_y] = vector_to_save
            df.loc[song_y, song_x] = vector_to_save

    df = df.sort_index(axis=0, ascending=True)
    df = df.sort_index(axis=1, ascending=True)
    return df
