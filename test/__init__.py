import os
import warnings
import configparser
from foucluster.transform import all_songs
from foucluster.distance import distance_matrix, distance_dict


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

# WAV
encoder = config['WAV']['encoder']
mp = True if str(config['WAV']['multiprocess']) == 'True' else False

# Fourier
rate_limit = float(config['Fourier']['rate'])
warp = config['Fourier']['warp']
warp = None if str(warp) == 'None' else int(warp)
step = float(config['Fourier']['step'])

all_songs(source_folder=source_folder,
          output_folder=output_folder,
          temp_folder=temp_folder,
          rate_limit=rate_limit,
          overwrite=False,
          plot=False,
          multiprocess=mp,
          encoder=encoder,
          step=step)

# Distance metric
metric = 'l2_norm'
song_df = distance_matrix(fourier_folder=output_folder,
                          warp=warp,
                          upper_limit=rate_limit,
                          distance_metric=metric)

song_df.to_csv(os.path.join(distance_folder,
                            metric + '.csv'),
               sep=';')
