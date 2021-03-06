import os
import configparser
from foucluster.structure import save_data
import pandas as pd
import unittest
from foucluster.transform import transform_folder
from foucluster.distance import distance_matrix, distance_dict
from foucluster.plot import heatmap_song
from foucluster.cluster import \
    score_cluster, n_cluster_methods, determinist_cluster
import warnings


class TestFullExample(unittest.TestCase):

    def test_example(self):
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

        # Fourier
        mp_w = True if str(config['Fourier']['multiprocess']) == 'True' else False
        rate_limit = float(config['Fourier']['rate'])
        step = float(config['Fourier']['step'])

        # Distance
        frames = int(config['Distance']['frames'])
        mp_d = True if str(config['Distance']['multiprocess']) == 'True' else False

        print('Transforming MP3 song into Fourier series...')
        transform_folder(source_folder=source_folder,
                         output_folder=output_folder,
                         temp_folder=temp_folder,
                         rate_limit=rate_limit,
                         overwrite=False,
                         plot=False,
                         image_folder=image_folder,
                         multiprocess=mp_w,
                         encoder=encoder,
                         step=step)

        # Distance metric
        print('Calculating distance matrix...')
        metrics = distance_dict.keys()
        song_data_dict = {}
        for metric in metrics:
            print(' ', metric)
            song_data = distance_matrix(fourier_folder=output_folder,
                                        multiprocess=mp_d,
                                        frames=frames,
                                        distance_metric=metric)
            song_data_dict[metric] = song_data

            save_data(song_data, folder=distance_folder, name=metric)

        # Clustering test
        print('Testing cluster methods...')
        max_score = 0.0
        score_vector = []
        metric_vector = []
        cluster_method_vector = []

        for metric in metrics:
            print(' ', metric)
            song_df = song_data_dict[metric]
            n_genres = 2
            for cluster_method in n_cluster_methods:
                cluster_df = determinist_cluster(dist_df=song_df.copy(deep=True),
                                                 method=cluster_method,
                                                 n_clusters=n_genres)
                score = score_cluster(cluster_df)
                cluster_df.to_csv(os.path.join(cluster_folder,
                                               metric + '_' +
                                               cluster_method +
                                               '.csv'),
                                  sep=';')
                print('  ', cluster_method, score)
                # Update info
                score_vector.append(score)
                metric_vector.append(metric)
                cluster_method_vector.append(cluster_method)
                # Choosing best methodology
                if score > max_score:
                    # print(metric, cluster_method, score)
                    max_score = score
                    best_metric = metric
                    best_cluster_method = cluster_method

        test_dict = {'Accuracy': score_vector,
                     'Metric': metric_vector,
                     'Cluster_method': cluster_method_vector}
        df = pd.DataFrame(test_dict)
        df.to_csv(os.path.join(cluster_folder,
                               'cluster_test.csv'),
                  sep=';', index=False)

        msg = 'Best performance ({}) is achieved with {} metric, {} cluster method'
        print(msg.format(max_score, best_metric, best_cluster_method))
        self.assertGreaterEqual(max_score, 0.9)


if __name__ == '__main__':
    unittest.main()
