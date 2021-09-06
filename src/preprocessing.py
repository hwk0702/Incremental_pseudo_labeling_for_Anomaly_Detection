import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import copy
import sys
from torch.utils.data import  Dataset, DataLoader
import torch

def group_by(data: pd.DataFrame, keys):
    """
    data grouping

    Arguments
    ---------
    data: dataframe
    keys: mapping keys

    """
    return data.groupby(keys)

class normalize:
    """
        data normalize

        Arguments
        ---------
        data

    """
    def __init__(self, data):
        self.min_max_scaler = MinMaxScaler()
        self._fit(data)

    def _fit(self, data):
        self.min_max_scaler.fit(data)

    def transform(self, data):
        return self.min_max_scaler.transform(data)


# def normalize(data: pd.DataFrame, var):
#     """
#     data normalize
#
#     Arguments
#     ---------
#     data: dataframe
#     var: Required variable
#
#     """
#     min_max_scaler = MinMaxScaler()
#     fitted = min_max_scaler.fit(data[var])
#     output = min_max_scaler.transform(data[var])
#     norm = copy.copy(data)
#     norm[var] = pd.DataFrame(output, columns=var, index=list(norm.index.values))
#     return norm


# def re_labeling(group_dataset, var, length, stand, flatten=False, stat=False):
#     """
#     relabeling
#
#     Arguments
#     ---------
#     data: dataframe
#     var: Required variable
#     length: Minimum length to be cut
#     stand: Number of particles to label(1 or 5)
#     stat: Whether to calculate the statistics or not
#
#     """
#     assert stand in [1, 5]
#     x = copy.copy(group_dataset[var].iloc[:length])
#     y = group_dataset['id'].iloc[:length][0]
#     y = np.array(y, dtype=np.int)
#     y[y < stand] = 0
#     y[y >= stand] = 1
#     if stat:
#         x = copy.copy(group_dataset[var[:-4]].iloc[:length])
#         x = x.describe().loc[['mean', 'std', 'min', 'max']]
#         x = x.append(x.median().rename('median'))
#     if flatten:
#         x = np.transpose(np.array(x)).flatten()
#     x = np.array(x, dtype=np.float64)
#     return x, y


def re_labeling(label, data):
    """
    relabeling

    Arguments
    ---------
    data: dataframe
    var: Required variable
    length: Minimum length to be cut
    stand: Number of particles to label(1 or 5)
    stat: Whether to calculate the statistics or not

    """    
    #isMinist check
    #if(!isMnist):
    #    return
    data['label'] = data.apply(lambda x : 1 if x['label'] == label else 0, axis =1)    
    return data

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.tensor(np.array(y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]