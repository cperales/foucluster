import json
import glob
import os
import numpy as np
from .structure import Data
from .transform import dict_to_array
from itertools import combinations_with_replacement
import multiprocessing as mp
import copy
# sqrt(2) with default precision np.float64
_SQRT2 = np.sqrt(2)


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


def warp_distance(distance_metric, features_x, y, warp=200):
    """
    Calculate the minimum distance among
    features_x and y arrays after warping.

    :param str distance_metric:
    :param np.array features_x:
    :param np.array y:
    :param int warp:
    :return:
    """
    # Selecting the array
    distance_func = distance_dict[distance_metric]
    # Copying the value
    features_x_copy = copy.deepcopy(features_x)
    y_copy = copy.deepcopy(y)
    # Starting the warping
    min_diff = distance_func(features_x, y)
    for i in range(1, int(warp)):
        # Moving forward
        forward_diff = distance_func(features_x_copy[i:], y_copy[:-i])
        if forward_diff < min_diff:
            min_diff = forward_diff
        # Moving backward
        backward_diff = distance_func(features_x_copy[:-i], y_copy[i:])
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

    if np.isnan(distance_array).any():
        raise ValueError('Error in distances')
    return distance_array


def distance_matrix(fourier_folder: str,
                    multiprocess: bool = False,
                    frames: int =1,
                    distance_metric: str ='l2_norm'):
    """
    A distance matrix with all the songs of a folder
    can be calculated.

    :param fourier_folder:
    :param int frames:
    :param distance_metric:
    :param bool multiprocess:
    :param str distance_metric:
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
    file_names = list(merged_file.keys())
    # song_names = [song.split('.')[-2] for song in list(merged_file.keys())]
    song_names = file_names
    data = Data(columns=song_names, shape=frames)

    if multiprocess is True:
        ff_dict = {}
        for song_name, file_name in zip(song_names, file_names):
            freq, features = dict_to_array(merged_file[file_name])
            ff_dict.update({song_name: {'freq': freq, 'features': features}})

        mgr = mp.Manager()
        ns = mgr.Namespace()
        ns.distance_metric = distance_metric
        ns.ff_dict = ff_dict
        ns.frames = frames
        # Distances are saved in a shared dict
        shared_dict = mgr.dict()
        for song_name in song_names:
            shared_dict[song_name] = mgr.dict()
        ns.dict = shared_dict

        # Args must be in list
        song_names_tuple = [comb for comb in combinations_with_replacement(song_names, r=2)]
        args_to_mp = [(names[0], names[1], ns) for names in song_names_tuple]

        with mp.Pool(processes=max(mp.cpu_count(), 1)) as p:
            p.starmap(multiprocess_matrix, args_to_mp)

        # Retrieve the information and save into the dataframe
        for k_1 in ns.dict.keys():
            for k_2 in ns.dict.keys():
                data.loc(k_1, k_2, ns.dict[k_1][k_2])
    else:
        for i in range(len(song_names)):
            for j in range(i, len(song_names)):
                song_x = song_names[i]
                if j > i:
                    # Song_x
                    freq_x, features_x = dict_to_array(merged_file[song_x])
                    song_y = song_names[j]
                    freq_y, features_y = dict_to_array(merged_file[song_y])
                    distance = pair_distance(freq_x=freq_x,
                                             features_x=features_x,
                                             freq_y=freq_y,
                                             features_y=features_y,
                                             frames=frames,
                                             distance_metric=distance_metric)
                    data.loc(song_x, song_y, distance)
                    # Save also in reverse
                    data.loc(song_y, song_x, distance)
                else:
                    data.loc(song_x, song_x, np.zeros(frames))
    return data


def multiprocess_matrix(song_x, song_y, ns):
    """

    :param song_x:
    :param song_y:
    :param ns: Namespace.
    :return:
    """
    if song_x == song_y:
        ns.dict[song_x][song_x] = np.zeros(ns.frames)
    else:
        # Song_x
        freq_x = ns.ff_dict[song_x]['freq']
        features_x = ns.ff_dict[song_x]['features']
        # Song_y
        freq_y = ns.ff_dict[song_y]['freq']
        features_y = ns.ff_dict[song_y]['features']
        # Distance
        distance = pair_distance(freq_x=freq_x,
                                 features_x=features_x,
                                 freq_y=freq_y,
                                 features_y=features_y,
                                 frames=ns.frames,
                                 distance_metric=ns.distance_metric)
        ns.dict[song_x][song_y] = distance
        # Save also in reverse
        ns.dict[song_y][song_x] = distance
