import os
import configparser
import numpy as np
import pandas as pd
from foucluster.transform import all_songs
from foucluster.distance import distance_matrix, distance_dict
from foucluster.plot import heatmap_song
from foucluster.cluster import \
    score_cluster, non_n_cluster_methods, determinist_cluster, automatic_cluster
import warnings
import logging

logger = logging.getLogger('foucluster')
logger.setLevel('DEBUG')

warnings.simplefilter("ignore")
config = configparser.ConfigParser()
config.read('config.ini')

# Folder
for folder in config['Folder'].values():
    os.makedirs(folder, exist_ok=True)
source_folder = config['Folder']['Source']
temp_folder = config['Folder']['Temp']
image_folder = config['Folder']['Image']
output_folder = config['Folder']['Output']
distance_folder = config['Folder']['Distance']
cluster_folder = config['Folder']['Cluster']

# WAV
encoder = config['WAV']['encoder']
mp = True if str(config['WAV']['multiprocess']) == 'True' else False
warp = config['Distance']['warp']
warp = None if str(warp) == 'None' else int(warp)

# Fourier
rate_limit = float(config['Fourier']['rate'])
step = float(config['Fourier']['step'])

metrics = distance_dict.keys()

logger.info('Transforming MP3 songs into Fourier series...')
all_songs(source_folder=source_folder,
          output_folder=output_folder,
          temp_folder=temp_folder,
          rate_limit=rate_limit,
          overwrite=False,
          plot=False,
          image_folder=image_folder,
          multiprocess=mp,
          encoder=encoder,
          step=step)

# Distance metric
print('Calculating distance matrix...')
song_distance = distance_matrix(fourier_folder=output_folder,
                                multiprocess=False,
                                distance_metric='positive')

song_pd = song_distance.to_df()

# # Heat map
# print('Plotting heat maps...')
# for metric in metrics:
#     print(' ', metric)
#     dist_df = pd.read_csv(os.path.join(distance_folder,
#                                        metric + '.csv'),
#                           sep=';')
#     dist_df = dist_df.set_index('Songs')
#     heatmap_song(dist_df,
#                  image_name=metric,
#                  image_folder=image_folder)

# Clustering test
print('Testing cluster methods...')
for metric in metrics:
    print(' ', metric)
    song_df = pd.read_csv(os.path.join(distance_folder,
                                       metric + '.csv'),
                          sep=';')
    song_df = song_df.set_index('Songs')
    for cluster_method in non_n_cluster_methods:
        cluster_df = automatic_cluster(dist_df=song_df.copy(deep=True),
                                       method=cluster_method)
        cluster_df.to_csv(os.path.join(cluster_folder,
                                       metric + '_' +
                                       cluster_method +
                                       '.csv'),
                          sep=';')
