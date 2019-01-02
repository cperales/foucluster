import os
import glob
import json
import subprocess
import multiprocessing as mp
import numpy as np
from scipy.io.wavfile import read
from .plot import fourier_plot, song_plot
import logging


logger = logging.getLogger('foucluster')
logger.setLevel('DEBUG')


def mp3_to_wav(mp3_file, wav_file, encoder='mpg123'):
    """
    Transform mp3 file into wav format calling bash and using mpg123
    or ffmpeg.

    :param str mp3_file: path to the mp3 file.
    :param str wav_file: path to the new wav file.
    :param str encoder: Encode to use. It could be mpg123 or ffmpeg.
    :return:
    """
    if encoder == 'mpg123':
        bash_command = ['mpg123', '-w', wav_file, '--mono', mp3_file]
    else:
        bash_command = ['ffmpeg', '-i', mp3_file, wav_file]
    subprocess.run(bash_command)


def wav_to_fourier(wav_file,
                   rate_limit=6000.0,
                   step=1.0):
    """
    WAV file is loaded and transformed into Fourier Series.
    This Fourier Series is limited.

    :param str wav_file:
    :param float rate_limit:
    :param float step:
    :return:
    """
    rate, aud_data = read(wav_file)
    # Should be mono
    if len(aud_data) != len(aud_data.ravel()):
        aud_data = np.mean(aud_data, axis=1)

    # Zero padding
    len_data = aud_data.shape[0]
    channel_1 = np.zeros(2 ** (int(np.ceil(np.log2(len_data)))))
    channel_1[0:len_data] = aud_data

    # Fourier analysis
    fourier = np.abs(np.fft.fft(channel_1))
    freq = np.linspace(0, rate, fourier.shape[0])

    freq, fourier = limit_by_freq(freq,
                                  fourier,
                                  upper_limit=rate_limit)
    freq, fourier = group_by_freq(freq,
                                  fourier,
                                  step=step)
    # Max frequency should be 100.0
    a = np.max(np.abs(fourier)) / 100.0
    fourier = fourier / a

    return freq, fourier


def group_by_freq(freq, features, step=1.0):
    """

    :param freq:
    :param features:
    :param step:
    :return:
    """
    min_freq = int(np.min(freq))
    max_freq = int(np.max(freq))
    length = int((max_freq - min_freq) / step) + 1
    new_freq = np.empty(length, dtype=np.float)
    new_features = np.empty(length, dtype=np.float)
    i = 0
    for freq_i in np.arange(min_freq, max_freq, step):
        mask_1 = freq >= freq_i
        mask_2 = freq < freq_i + step
        mask = mask_1 * mask_2
        new_freq[i] = np.mean(freq[mask])
        new_features[i] = np.mean(features[mask])
        i += 1
    new_freq = np.array(new_freq, dtype=np.float)
    new_features = np.array(new_features, dtype=np.float)
    return new_freq, new_features


def limit_by_freq(freq, features, upper_limit, lower_limit=None):
    """
    Limit arrays of frequency and features by maximum frequency and
    bottom frequency.

    :param freq: array of frequencies.
    :param features: array of amplitude.
    :param float upper_limit: maximum frequency.
    :param float lower_limit: minimum frequency.
    :return:
    """
    # Copy into arrays, in order to apply mask
    freq = np.array(freq, dtype=np.float)
    features = np.array(features, dtype=np.float)
    # Mask for bottom limit
    if lower_limit is not None:
        bottom_mask = freq >= lower_limit
        features = features[bottom_mask]
        freq = freq[bottom_mask]
    # Mask for upper limit
    upper_mask = freq <= upper_limit
    features = features[upper_mask]
    freq = freq[upper_mask]
    return freq, features


def dict_to_array(song_dict):
    """

    :param dict song_dict: load form dictionary to array
    :return:
    """
    freq = np.array([k for k in song_dict.keys()], dtype=np.float)
    features = np.array([v for v in song_dict.values()], dtype=np.float)
    return freq, features


def check_wav(song, source_folder, temp_folder, encoder='mpg123'):
    """
    Check if song is already transformed into temp.

    :param str song:
    :param str source_folder:
    :param str temp_folder:
    :param str encoder:
    :return:
    """
    # Name of files
    song_name = os.path.splitext(song)[0]
    mp3_file = os.path.join(source_folder, song)
    wav_file = os.path.join(temp_folder, song_name + '.wav')
    try:
        if not os.path.isfile(wav_file):
            mp3_to_wav(
                mp3_file=mp3_file,
                wav_file=wav_file,
                encoder=encoder)
        else:
            pass
    except MemoryError:
        logger.error('MemoryError: %s MP3 couldn\'t be transformed into WAV', song_name)


def time_to_frequency(song,
                      temp_folder,
                      output_folder,
                      rate_limit=6000.0,
                      overwrite=True,
                      plot=True,
                      image_folder=None,
                      step=5.0):
    """
    Transform a MP3 song into WAV format, and then into
    Fourier series.

    :param str song: name of the song, with MP3 extension.
    :param str output_folder: folder where pickle files from
        frequency series are saved.
    :param str temp_folder: folder where wav files are saved.
    :param float rate_limit: maximum frequency of the frequency
        series.
    :param bool overwrite:
    :param bool plot: if True, frequency series is plotted.
    :param image_folder: if plotting is True, is the folder
        where the Fourier data is saved.
    :param float step: step of the Fourier series.
    :return:
    """
    # Name of files
    song_name = os.path.splitext(song)[0]
    json_name = os.path.join(output_folder, song_name + '.json')
    wav_file = os.path.join(temp_folder, song_name + '.wav')

    if not os.path.isfile(json_name) or overwrite is True:
        # Fourier transformation
        try:
            frequencies, fourier_series = wav_to_fourier(wav_file=wav_file,
                                                         rate_limit=rate_limit,
                                                         step=step)

            # Save as JSON
            json_to_save = {song: {str(x): y for x, y in
                                   zip(frequencies, fourier_series)}}
            with open(json_name, 'w') as output:
                json.dump(json_to_save, output)

            # Plotting
            if plot is True:
                fourier_plot(freq=frequencies,
                             features=fourier_series,
                             folder=image_folder,
                             filename=song_name)
            logger.debug('%s transformed', song_name)
        except MemoryError:
            logger.error('MemoryError: %s couldn\'t be Fourier transformed', song_name)


def all_songs(source_folder,
              output_folder,
              temp_folder,
              rate_limit=6000.0,
              overwrite=True,
              plot=False,
              image_folder=None,
              multiprocess=False,
              encoder='mpg123',
              step=5.0):
    """
    Transform a directory full of MP3 files
    into WAV files, and then into Fourier series,
    working with directories.

    :param str source_folder: folder where MP3 files are.
    :param str output_folder: folder where pickle files from
        frequency series are saved.
    :param str temp_folder: folder where wav files are saved.
    :param float rate_limit: maximum frequency of the frequency
        series.
    :param bool overwrite:
    :param bool plot: if True, frequency series is plotted.
    :param image_folder: if plotting is True, is the folder
        where the Fourier data is saved.
    :param bool multiprocess: if True, encoding and Fourier transform
        are run in several cores. It may be unstable (consume to much RAM).
    :param str encoder: encoder from MP3 to WAV.
    :param float step: step of the Fourier series.
    """
    merged_file = os.path.join(output_folder, 'merged_file.json')

    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    if os.path.isfile(merged_file):
        os.remove(merged_file)
    if plot:
        os.makedirs(image_folder, exist_ok=True)

    # Check if mp3 is already transformed into wav. Right
    # now, foucluster doesn't have a direct read from mp3
    logger.info('Checking if songs are in WAV format...')
    [check_wav(song=song,
               source_folder=source_folder,
               temp_folder=temp_folder,
               encoder=encoder)
     for song in os.listdir(source_folder)]

    if multiprocess is True:
        logger.debug('Fourier is applied in multiprocess')
        songs = [(song, temp_folder, output_folder, rate_limit,
                  overwrite, plot, image_folder, step)
                 for song in os.listdir(source_folder)]

        # with mp.Pool(processes=max(int(mp.cpu_count() / 2.0), 1)) as p:
        with mp.Pool(processes=max(int(mp.cpu_count() - 1), 1)) as p:
            p.starmap(time_to_frequency, songs)
    else:
        logger.debug('Fourier is applied in single core')
        [time_to_frequency(song=song,
                           temp_folder=temp_folder,
                           output_folder=output_folder,
                           rate_limit=rate_limit,
                           overwrite=overwrite,
                           plot=plot,
                           image_folder=image_folder,
                           step=step)
         for song in os.listdir(source_folder)]

    read_files = glob.glob(os.path.join(output_folder, '*.json'))

    with open(merged_file, 'w') as outfile:
        file_contents = [open(f).read() for f in read_files]
        outfile.write('[{}]'.format(','.join(file_contents)))
