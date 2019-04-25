import copy
import os
import numpy as np
import pandas as pd


class Data:
    """
    Dummy class in order to store into the dataframe.
    """

    def __init__(self, columns, shape):
        self.columns = columns
        self.index = columns
        self.dict_ = {c_1: {c_2: [] for c_2 in columns}
                      for c_1 in columns}
        self.shape = shape

    def loc(self, pos_x, pos_y, vector):
        """

        :param pos_x:
        :param pos_y:
        :param vector:
        :return:
        """
        self.dict_[pos_x][pos_y] = vector
        self.dict_[pos_y][pos_x] = self.dict_[pos_x][pos_y]

    def copy(self, deep=True):
        """

        :param deep:
        :return:
        """
        if deep is True:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def to_df(self):
        """
        Export data as a pandas.DataFrame.

        :return:
        """
        if self.shape > 1:
            range_str = [s for s in range(self.shape)]
            iterables = [self.columns, range_str]
            multiindex = pd.MultiIndex.from_product(iterables, names=['song', 'frame'])
            # multiindex = [i for i in itertools.product(self.columns, range_str, repeat=1)]
            df = pd.DataFrame(columns=multiindex, index=self.columns, dtype=np.float64)

            for c_1 in self.columns:
                for c_2 in self.columns:
                    for s in range_str:
                        df.loc[c_1][c_2, s] = self.dict_[c_1][c_2][s]
            df = df.T
        else:
            df = pd.DataFrame(columns=self.columns + ['song'], dtype=np.float64)
            df['song'] = self.columns
            df = df.set_index('song')

            for c_1 in self.columns:
                for c_2 in self.columns:
                    df.loc[c_1, c_2] = self.max_diff(c_1, c_2)

        return df

    # TODO
    def to_json(self):
        """
        Export data as a JSON.

        :return:
        """
        return None

    def min_diff(self, song_x, song_y):
        """

        :param song_x:
        :param song_y:
        :return:
        """
        array = self.dict_[song_x][song_y]
        return np.min(array)

    def max_diff(self, song_x, song_y):
        """

        :param song_x:
        :param song_y:
        :return:
        """
        array = self.dict_[song_x][song_y]
        return np.max(array)

    def pos_diff(self, song_x, song_y, pos):
        """

        :param song_x:
        :param song_y:
        :param pos:
        :return:
        """
        array = self.dict_[song_x][song_y]
        return array[pos]


def save_data(data: Data, folder, name):
    """
    Function to save data into csv.

    :param Data data:
    :param str folder:
    :param str name:
    :return:
    """
    if data.shape == 1:
        data.to_df().to_csv(os.path.join(folder,
                                         '.'.join([name, 'csv'])),
                            sep=';', index_label='song')
    else:
        data.to_df().to_csv(os.path.join(folder,
                                         '.'.join([name, 'csv'])),
                            sep=';',
                            index_label=['song', 'frame'])
