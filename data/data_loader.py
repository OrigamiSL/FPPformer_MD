import os
import warnings
import copy
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from utils.modwt import modwt, modwtmra
from utils.DMD import DMD_reconstruct
from utils.StiefelGen import StiefelGen
from utils.FOC import FOC
import time
import matplotlib.pyplot as plt
from astropy.stats import biweight_midvariance

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ECL.csv',
                 Batch_size=16, MODWT_level=5,
                 augmentation_method='DMD', augmentation_ratio=0.5, augmentation_len=48):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.MODWT_level = MODWT_level  # MODWT_level
        self.eps = 1e-5  # eps
        self.batch_size = Batch_size
        self.arg_m = augmentation_method
        self.aug_r = augmentation_ratio
        self.aug_l = augmentation_len
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.input_len, 12 * 30 * 24 + 4 * 30 * 24 - self.input_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.data = self.scaler.transform(df_value)
        self.data_x = self.data[border1:border2]
        self.window_num = self.data_x.shape[0] - self.input_len - self.pred_len + 1
        if self.set_type == 0:
            self.index_list = np.arange(self.window_num)
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def __getitem__(self, index):
        if self.set_type == 0:
            seq_x = []
            seq_x_mra = []
            var_speriods = []
            var_sp_matrix = []
            num_list = np.arange(self.data_x.shape[1])
            num = self.data_x.shape[1]
            for idx in self.index_list[index * self.batch_size: (index + 1) * self.batch_size]:
                current_index = np.random.choice(num_list, num, replace=False)
                r_begin = idx
                r_end = r_begin + self.input_len + self.pred_len
                seq_x_temp = copy.deepcopy(self.data_x[r_begin:r_end, current_index])
                seq_x.append(seq_x_temp)

                input_x_temp = seq_x_temp[:self.input_len, :]
                wp = modwt(input_x_temp, 'db4', self.MODWT_level)
                mra = modwtmra(wp, 'db4')
                input_x_mra_temp = np.cumsum(np.flip(mra, axis=0), axis=0)
                seq_x_mra.append(input_x_mra_temp[:-1])

                bivar = biweight_midvariance(wp, axis=1)
                var_speriod = np.argmax(bivar, axis=0)
                var_speriods.append(var_speriod)
                ones_matrix = np.ones([1, input_x_temp.shape[1]])
                zeros_matrix = np.zeros([1, input_x_temp.shape[1]])
                var_sp_matrix_temp = np.ones([input_x_temp.shape[1], input_x_temp.shape[1]])
                for i in range(input_x_temp.shape[1]):
                    shared_period = var_speriod[i]
                    if shared_period == 0:
                        var_sp_matrix_temp[i, i] = 0
                    else:
                        var_sp_matrix_temp[i, :] = np.where(var_speriod == shared_period, zeros_matrix, ones_matrix)
                var_sp_matrix.append(var_sp_matrix_temp)
            seq_x = np.stack(seq_x, axis=0)

            var_speriods = np.stack(var_speriods, axis=0)
            per, per_num = np.unique(var_speriods, return_counts=True)
            sper = per[np.argmax(per_num)]
            sper_index = np.where(var_speriods == sper)

            start_index = np.random.randint(self.input_len - self.aug_l - 1)
            seq_x_sper = seq_x[sper_index[0], start_index:start_index + self.aug_l + 1, sper_index[1]]
            if seq_x_sper.shape[0] > 2 * self.aug_l and np.random.rand() < self.aug_r:
                if self.arg_m == 'DMD':
                    seq_x[sper_index[0], start_index:start_index + self.aug_l, sper_index[1]] = (
                        DMD_reconstruct(seq_x_sper, self.aug_l, 1))
                elif self.arg_m == 'StiefelGen':
                    seq_x[sper_index[0], start_index:start_index + self.aug_l, sper_index[1]] = (
                        StiefelGen(seq_x_sper)[:, :-1])
                elif self.arg_m == 'FOC':
                    seq_x[sper_index[0], start_index:start_index + self.aug_l, sper_index[1]] = (
                        FOC(seq_x_sper[:, :-1]))

            seq_x_mra = np.stack(seq_x_mra, axis=0)
            var_sp_matrix = np.stack(var_sp_matrix, axis=0)
        else:
            r_begin = index
            r_end = r_begin + self.input_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]
            input_x = seq_x[:self.input_len, :]
            wp = modwt(input_x, 'db4', self.MODWT_level)
            mra = modwtmra(wp, 'db4')
            seq_x_mra = np.cumsum(np.flip(mra, axis=0), axis=0)[:-1]
            bivar = biweight_midvariance(wp, axis=1)
            var_speriod = np.argmax(bivar, axis=0)
            ones_matrix = np.ones([1, input_x.shape[1]])
            zeros_matrix = np.zeros([1, input_x.shape[1]])
            var_sp_matrix = np.ones([input_x.shape[1], input_x.shape[1]])
            for i in range(input_x.shape[1]):
                shared_period = var_speriod[i]
                if shared_period == 0:
                    var_sp_matrix[i, i] = 0
                else:
                    var_sp_matrix[i, :] = np.where(var_speriod == shared_period, zeros_matrix, ones_matrix)

        return seq_x, seq_x_mra, var_sp_matrix

    def __len__(self):
        if self.set_type == 0:
            return (len(self.data_x) - self.input_len - self.pred_len + 1) // self.batch_size
        else:
            return len(self.data_x) - self.input_len - self.pred_len + 1

    def train_shuffle(self):
        if self.set_type == 0:
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_min(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ECL.csv',
                 Batch_size=16, MODWT_level=5,
                 augmentation_method='DMD', augmentation_ratio=0.5, augmentation_len=48):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.MODWT_level = MODWT_level  # MODWT_level
        self.eps = 1e-5  # eps
        self.batch_size = Batch_size
        self.arg_m = augmentation_method
        self.aug_r = augmentation_ratio
        self.aug_l = augmentation_len
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.input_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.input_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.data = self.scaler.transform(df_value)
        self.data_x = self.data[border1:border2]
        self.window_num = self.data_x.shape[0] - self.input_len - self.pred_len + 1
        if self.set_type == 0:
            self.index_list = np.arange(self.window_num)
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def __getitem__(self, index):
        if self.set_type == 0:
            seq_x = []
            seq_x_mra = []
            var_speriods = []
            var_sp_matrix = []
            num_list = np.arange(self.data_x.shape[1])
            num = self.data_x.shape[1]
            for idx in self.index_list[index * self.batch_size: (index + 1) * self.batch_size]:
                current_index = np.random.choice(num_list, num, replace=False)
                r_begin = idx
                r_end = r_begin + self.input_len + self.pred_len
                seq_x_temp = copy.deepcopy(self.data_x[r_begin:r_end, current_index])
                seq_x.append(seq_x_temp)

                input_x_temp = seq_x_temp[:self.input_len, :]
                wp = modwt(input_x_temp, 'db4', self.MODWT_level)
                mra = modwtmra(wp, 'db4')
                input_x_mra_temp = np.cumsum(np.flip(mra, axis=0), axis=0)
                seq_x_mra.append(input_x_mra_temp[:-1])

                bivar = biweight_midvariance(wp, axis=1)
                var_speriod = np.argmax(bivar, axis=0)
                var_speriods.append(var_speriod)
                ones_matrix = np.ones([1, input_x_temp.shape[1]])
                zeros_matrix = np.zeros([1, input_x_temp.shape[1]])
                var_sp_matrix_temp = np.ones([input_x_temp.shape[1], input_x_temp.shape[1]])
                for i in range(input_x_temp.shape[1]):
                    shared_period = var_speriod[i]
                    if shared_period == 0:
                        var_sp_matrix_temp[i, i] = 0
                    else:
                        var_sp_matrix_temp[i, :] = np.where(var_speriod == shared_period, zeros_matrix, ones_matrix)
                var_sp_matrix.append(var_sp_matrix_temp)
            seq_x = np.stack(seq_x, axis=0)

            var_speriods = np.stack(var_speriods, axis=0)
            per, per_num = np.unique(var_speriods, return_counts=True)
            sper = per[np.argmax(per_num)]
            sper_index = np.where(var_speriods == sper)

            start_index = np.random.randint(self.input_len - self.aug_l - 1)
            seq_x_sper = seq_x[sper_index[0], start_index:start_index + self.aug_l + 1, sper_index[1]]
            if seq_x_sper.shape[0] > 2 * self.aug_l and np.random.rand() < self.aug_r:
                if self.arg_m == 'DMD':
                    seq_x[sper_index[0], start_index:start_index + self.aug_l, sper_index[1]] = (
                        DMD_reconstruct(seq_x_sper, self.aug_l, 1))
                elif self.arg_m == 'StiefelGen':
                    seq_x[sper_index[0], start_index:start_index + self.aug_l, sper_index[1]] = (
                        StiefelGen(seq_x_sper)[:, :-1])
                elif self.arg_m == 'FOC':
                    seq_x[sper_index[0], start_index:start_index + self.aug_l, sper_index[1]] = (
                        FOC(seq_x_sper[:, :-1]))

            seq_x_mra = np.stack(seq_x_mra, axis=0)
            var_sp_matrix = np.stack(var_sp_matrix, axis=0)
        else:
            r_begin = index
            r_end = r_begin + self.input_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]
            input_x = seq_x[:self.input_len, :]
            wp = modwt(input_x, 'db4', self.MODWT_level)
            mra = modwtmra(wp, 'db4')
            seq_x_mra = np.cumsum(np.flip(mra, axis=0), axis=0)[:-1]
            bivar = biweight_midvariance(wp, axis=1)
            var_speriod = np.argmax(bivar, axis=0)
            ones_matrix = np.ones([1, input_x.shape[1]])
            zeros_matrix = np.zeros([1, input_x.shape[1]])
            var_sp_matrix = np.ones([input_x.shape[1], input_x.shape[1]])
            for i in range(input_x.shape[1]):
                shared_period = var_speriod[i]
                if shared_period == 0:
                    var_sp_matrix[i, i] = 0
                else:
                    var_sp_matrix[i, :] = np.where(var_speriod == shared_period, zeros_matrix, ones_matrix)

        return seq_x, seq_x_mra, var_sp_matrix

    def __len__(self):
        if self.set_type == 0:
            return (len(self.data_x) - self.input_len - self.pred_len + 1) // self.batch_size
        else:
            return len(self.data_x) - self.input_len - self.pred_len + 1

    def train_shuffle(self):
        if self.set_type == 0:
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ECL.csv',
                 Batch_size=16, MODWT_level=3,
                 augmentation_method='DMD', augmentation_ratio=0.5, augmentation_len=48):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.MODWT_level = MODWT_level  # MODWT_level
        self.eps = 1e-5  # eps
        self.batch_size = Batch_size
        self.arg_m = augmentation_method
        self.aug_r = augmentation_ratio
        self.aug_l = augmentation_len
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.input_len, len(df_raw) - num_test - self.input_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.data = self.scaler.transform(df_value)
        self.data_x = self.data[border1:border2]
        self.window_num = self.data_x.shape[0] - self.input_len - self.pred_len + 1
        if self.set_type == 0:
            self.index_list = np.arange(self.window_num)
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def __getitem__(self, index):
        if self.set_type == 0:
            seq_x = []
            seq_x_mra = []
            var_speriods = []
            var_sp_matrix = []
            num_list = np.arange(self.data_x.shape[1])
            if self.data_x.shape[1] > 100:
                num = np.random.randint(low=5 * int(np.log(self.data_x.shape[1])),
                                        high=min(
                                            max(10 * int(np.log(self.data_x.shape[1])), self.data_x.shape[1] // 4),
                                            self.data_x.shape[1]),
                                        size=1)
            else:
                num = self.data_x.shape[1]
            for idx in self.index_list[index * self.batch_size: (index + 1) * self.batch_size]:
                current_index = np.random.choice(num_list, num, replace=False)
                r_begin = idx
                r_end = r_begin + self.input_len + self.pred_len
                seq_x_temp = copy.deepcopy(self.data_x[r_begin:r_end, current_index])
                seq_x.append(seq_x_temp)

                input_x_temp = seq_x_temp[:self.input_len, :]
                wp = modwt(input_x_temp, 'db4', self.MODWT_level)
                mra = modwtmra(wp, 'db4')
                input_x_mra_temp = np.cumsum(np.flip(mra, axis=0), axis=0)
                seq_x_mra.append(input_x_mra_temp[:-1])

                bivar = biweight_midvariance(wp, axis=1)
                var_speriod = np.argmax(bivar, axis=0)
                var_speriods.append(var_speriod)
                ones_matrix = np.ones([1, input_x_temp.shape[1]])
                zeros_matrix = np.zeros([1, input_x_temp.shape[1]])
                var_sp_matrix_temp = np.ones([input_x_temp.shape[1], input_x_temp.shape[1]])
                for i in range(input_x_temp.shape[1]):
                    shared_period = var_speriod[i]
                    if shared_period == 0:
                        var_sp_matrix_temp[i, i] = 0
                    else:
                        var_sp_matrix_temp[i, :] = np.where(var_speriod == shared_period, zeros_matrix, ones_matrix)
                var_sp_matrix.append(var_sp_matrix_temp)
            seq_x = np.stack(seq_x, axis=0)

            var_speriods = np.stack(var_speriods, axis=0)
            per, per_num = np.unique(var_speriods, return_counts=True)
            sper = per[np.argmax(per_num)]
            sper_index = np.where(var_speriods == sper)

            start_index = np.random.randint(self.input_len - self.aug_l - 1)
            seq_x_sper = seq_x[sper_index[0], start_index:start_index + self.aug_l + 1, sper_index[1]]
            if seq_x_sper.shape[0] > 2 * self.aug_l and np.random.rand() < self.aug_r:
                if self.arg_m == 'DMD':
                    seq_x[sper_index[0], start_index:start_index + self.aug_l, sper_index[1]] = (
                        DMD_reconstruct(seq_x_sper, self.aug_l, 1))
                elif self.arg_m == 'StiefelGen':
                    seq_x[sper_index[0], start_index:start_index + self.aug_l, sper_index[1]] = (
                        StiefelGen(seq_x_sper)[:, :-1])
                elif self.arg_m == 'FOC':
                    seq_x[sper_index[0], start_index:start_index + self.aug_l, sper_index[1]] = (
                        FOC(seq_x_sper[:, :-1]))

            seq_x_mra = np.stack(seq_x_mra, axis=0)
            var_sp_matrix = np.stack(var_sp_matrix, axis=0)
        else:
            r_begin = index
            r_end = r_begin + self.input_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]
            input_x = seq_x[:self.input_len, :]
            wp = modwt(input_x, 'db4', self.MODWT_level)
            mra = modwtmra(wp, 'db4')
            seq_x_mra = np.cumsum(np.flip(mra, axis=0), axis=0)[:-1]
            bivar = biweight_midvariance(wp, axis=1)
            var_speriod = np.argmax(bivar, axis=0)
            ones_matrix = np.ones([1, input_x.shape[1]])
            zeros_matrix = np.zeros([1, input_x.shape[1]])
            var_sp_matrix = np.ones([input_x.shape[1], input_x.shape[1]])
            for i in range(input_x.shape[1]):
                shared_period = var_speriod[i]
                if shared_period == 0:
                    var_sp_matrix[i, i] = 0
                else:
                    var_sp_matrix[i, :] = np.where(var_speriod == shared_period, zeros_matrix, ones_matrix)

        return seq_x, seq_x_mra, var_sp_matrix

    def __len__(self):
        if self.set_type == 0:
            return (len(self.data_x) - self.input_len - self.pred_len + 1) // self.batch_size
        else:
            return len(self.data_x) - self.input_len - self.pred_len + 1

    def train_shuffle(self):
        if self.set_type == 0:
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# https://github.com/guoshnBJTU/ASTGNN/tree/main/data
class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='PEMS08.npz',
                 Batch_size=16, MODWT_level=3,
                 augmentation_method='DMD', augmentation_ratio=0.5, augmentation_len=48):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.MODWT_level = MODWT_level  # MODWT_level
        self.eps = 1e-5  # eps
        self.batch_size = Batch_size
        self.arg_m = augmentation_method
        self.aug_r = augmentation_ratio
        self.aug_l = augmentation_len
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = np.load(os.path.join(self.root_path, self.data_path), allow_pickle=True)
        df_value = df_raw['data'][:, :, 0]
        num_train = int(len(df_value) * 0.7)
        num_test = int(len(df_value) * 0.2)
        num_vali = len(df_value) - num_train - num_test
        border1s = [0, num_train - self.input_len, len(df_value) - num_test - self.input_len]
        border2s = [num_train, num_train + num_vali, len(df_value)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_value

        # data standardization
        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_data)

        self.data_x = data[border1:border2]

        self.window_num = self.data_x.shape[0] - self.input_len - self.pred_len + 1
        if self.set_type == 0:
            self.index_list = np.arange(self.window_num)
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def __getitem__(self, index):
        if self.set_type == 0:
            seq_x = []
            seq_x_mra = []
            var_speriods = []
            var_sp_matrix = []
            num_list = np.arange(self.data_x.shape[1])
            if self.data_x.shape[1] > 100:
                num = np.random.randint(low=5 * int(np.log(self.data_x.shape[1])),
                                        high=min(
                                            max(10 * int(np.log(self.data_x.shape[1])), self.data_x.shape[1] // 4),
                                            self.data_x.shape[1]),
                                        size=1)
            else:
                num = self.data_x.shape[1]
            for idx in self.index_list[index * self.batch_size: (index + 1) * self.batch_size]:
                current_index = np.random.choice(num_list, num, replace=False)
                r_begin = idx
                r_end = r_begin + self.input_len + self.pred_len
                seq_x_temp = copy.deepcopy(self.data_x[r_begin:r_end, current_index])
                seq_x.append(seq_x_temp)

                input_x_temp = seq_x_temp[:self.input_len, :]
                wp = modwt(input_x_temp, 'db4', self.MODWT_level)
                mra = modwtmra(wp, 'db4')
                input_x_mra_temp = np.cumsum(np.flip(mra, axis=0), axis=0)
                seq_x_mra.append(input_x_mra_temp[:-1])

                bivar = biweight_midvariance(wp, axis=1)
                var_speriod = np.argmax(bivar, axis=0)
                var_speriods.append(var_speriod)
                ones_matrix = np.ones([1, input_x_temp.shape[1]])
                zeros_matrix = np.zeros([1, input_x_temp.shape[1]])
                var_sp_matrix_temp = np.ones([input_x_temp.shape[1], input_x_temp.shape[1]])
                for i in range(input_x_temp.shape[1]):
                    shared_period = var_speriod[i]
                    if shared_period == 0:
                        var_sp_matrix_temp[i, i] = 0
                    else:
                        var_sp_matrix_temp[i, :] = np.where(var_speriod == shared_period, zeros_matrix, ones_matrix)
                var_sp_matrix.append(var_sp_matrix_temp)
            seq_x = np.stack(seq_x, axis=0)

            var_speriods = np.stack(var_speriods, axis=0)
            per, per_num = np.unique(var_speriods, return_counts=True)
            sper = per[np.argmax(per_num)]
            sper_index = np.where(var_speriods == sper)

            start_index = np.random.randint(self.input_len - self.aug_l - 1)
            seq_x_sper = seq_x[sper_index[0], start_index:start_index + self.aug_l + 1, sper_index[1]]
            if seq_x_sper.shape[0] > 2 * self.aug_l and np.random.rand() < self.aug_r:
                if self.arg_m == 'DMD':
                    seq_x[sper_index[0], start_index:start_index + self.aug_l, sper_index[1]] = (
                        DMD_reconstruct(seq_x_sper, self.aug_l, 1))
                elif self.arg_m == 'StiefelGen':
                    seq_x[sper_index[0], start_index:start_index + self.aug_l, sper_index[1]] = (
                        StiefelGen(seq_x_sper)[:, :-1])
                elif self.arg_m == 'FOC':
                    seq_x[sper_index[0], start_index:start_index + self.aug_l, sper_index[1]] = (
                        FOC(seq_x_sper[:, :-1]))

            seq_x_mra = np.stack(seq_x_mra, axis=0)
            var_sp_matrix = np.stack(var_sp_matrix, axis=0)
        else:
            r_begin = index
            r_end = r_begin + self.input_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]
            input_x = seq_x[:self.input_len, :]
            wp = modwt(input_x, 'db4', self.MODWT_level)
            mra = modwtmra(wp, 'db4')
            seq_x_mra = np.cumsum(np.flip(mra, axis=0), axis=0)[:-1]
            bivar = biweight_midvariance(wp, axis=1)
            var_speriod = np.argmax(bivar, axis=0)
            ones_matrix = np.ones([1, input_x.shape[1]])
            zeros_matrix = np.zeros([1, input_x.shape[1]])
            var_sp_matrix = np.ones([input_x.shape[1], input_x.shape[1]])
            for i in range(input_x.shape[1]):
                shared_period = var_speriod[i]
                if shared_period == 0:
                    var_sp_matrix[i, i] = 0
                else:
                    var_sp_matrix[i, :] = np.where(var_speriod == shared_period, zeros_matrix, ones_matrix)

        return seq_x, seq_x_mra, var_sp_matrix

    def __len__(self):
        if self.set_type == 0:
            return (len(self.data_x) - self.input_len - self.pred_len + 1) // self.batch_size
        else:
            return len(self.data_x) - self.input_len - self.pred_len + 1

    def train_shuffle(self):
        if self.set_type == 0:
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# DCRNN
# https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX
class Dataset_PEMSBAY(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='pems-bay.h5',
                 Batch_size=16, MODWT_level=3,
                 augmentation_method='DMD', augmentation_ratio=0.5, augmentation_len=48):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.MODWT_level = MODWT_level  # MODWT_level
        self.eps = 1e-5  # eps
        self.batch_size = Batch_size
        self.arg_m = augmentation_method
        self.aug_r = augmentation_ratio
        self.aug_l = augmentation_len
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = h5py.File(os.path.join(self.root_path, self.data_path), 'r')
        df_value = df_raw['speed/block0_values']
        num_train = int(len(df_value) * 0.7)
        num_test = int(len(df_value) * 0.2)
        num_vali = len(df_value) - num_train - num_test
        border1s = [0, num_train - self.input_len, len(df_value) - num_test - self.input_len]
        border2s = [num_train, num_train + num_vali, len(df_value)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_value

        # data standardization
        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_data)
        self.data_x = data[border1:border2]

        self.window_num = self.data_x.shape[0] - self.input_len - self.pred_len + 1
        if self.set_type == 0:
            self.index_list = np.arange(self.window_num)
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def __getitem__(self, index):
        if self.set_type == 0:
            seq_x = []
            seq_x_mra = []
            var_speriods = []
            var_sp_matrix = []
            num_list = np.arange(self.data_x.shape[1])
            if self.data_x.shape[1] > 100:
                num = np.random.randint(low=5 * int(np.log(self.data_x.shape[1])),
                                        high=min(
                                            max(10 * int(np.log(self.data_x.shape[1])), self.data_x.shape[1] // 4),
                                            self.data_x.shape[1]),
                                        size=1)
            else:
                num = self.data_x.shape[1]
            for idx in self.index_list[index * self.batch_size: (index + 1) * self.batch_size]:
                current_index = np.random.choice(num_list, num, replace=False)
                r_begin = idx
                r_end = r_begin + self.input_len + self.pred_len
                seq_x_temp = copy.deepcopy(self.data_x[r_begin:r_end, current_index])
                seq_x.append(seq_x_temp)

                input_x_temp = seq_x_temp[:self.input_len, :]
                wp = modwt(input_x_temp, 'db4', self.MODWT_level)
                mra = modwtmra(wp, 'db4')
                input_x_mra_temp = np.cumsum(np.flip(mra, axis=0), axis=0)
                seq_x_mra.append(input_x_mra_temp[:-1])

                bivar = biweight_midvariance(wp, axis=1)
                var_speriod = np.argmax(bivar, axis=0)
                var_speriods.append(var_speriod)
                ones_matrix = np.ones([1, input_x_temp.shape[1]])
                zeros_matrix = np.zeros([1, input_x_temp.shape[1]])
                var_sp_matrix_temp = np.ones([input_x_temp.shape[1], input_x_temp.shape[1]])
                for i in range(input_x_temp.shape[1]):
                    shared_period = var_speriod[i]
                    if shared_period == 0:
                        var_sp_matrix_temp[i, i] = 0
                    else:
                        var_sp_matrix_temp[i, :] = np.where(var_speriod == shared_period, zeros_matrix, ones_matrix)
                var_sp_matrix.append(var_sp_matrix_temp)
            seq_x = np.stack(seq_x, axis=0)

            var_speriods = np.stack(var_speriods, axis=0)
            per, per_num = np.unique(var_speriods, return_counts=True)
            sper = per[np.argmax(per_num)]
            sper_index = np.where(var_speriods == sper)

            start_index = np.random.randint(self.input_len - self.aug_l - 1)
            seq_x_sper = seq_x[sper_index[0], start_index:start_index + self.aug_l + 1, sper_index[1]]
            if seq_x_sper.shape[0] > 2 * self.aug_l and np.random.rand() < self.aug_r:
                if self.arg_m == 'DMD':
                    seq_x[sper_index[0], start_index:start_index + self.aug_l, sper_index[1]] = (
                        DMD_reconstruct(seq_x_sper, self.aug_l, 1))
                elif self.arg_m == 'StiefelGen':
                    seq_x[sper_index[0], start_index:start_index + self.aug_l, sper_index[1]] = (
                        StiefelGen(seq_x_sper)[:, :-1])
                elif self.arg_m == 'FOC':
                    seq_x[sper_index[0], start_index:start_index + self.aug_l, sper_index[1]] = (
                        FOC(seq_x_sper[:, :-1]))

            seq_x_mra = np.stack(seq_x_mra, axis=0)
            var_sp_matrix = np.stack(var_sp_matrix, axis=0)
        else:
            r_begin = index
            r_end = r_begin + self.input_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]
            input_x = seq_x[:self.input_len, :]
            wp = modwt(input_x, 'db4', self.MODWT_level)
            mra = modwtmra(wp, 'db4')
            seq_x_mra = np.cumsum(np.flip(mra, axis=0), axis=0)[:-1]
            bivar = biweight_midvariance(wp, axis=1)
            var_speriod = np.argmax(bivar, axis=0)
            ones_matrix = np.ones([1, input_x.shape[1]])
            zeros_matrix = np.zeros([1, input_x.shape[1]])
            var_sp_matrix = np.ones([input_x.shape[1], input_x.shape[1]])
            for i in range(input_x.shape[1]):
                shared_period = var_speriod[i]
                if shared_period == 0:
                    var_sp_matrix[i, i] = 0
                else:
                    var_sp_matrix[i, :] = np.where(var_speriod == shared_period, zeros_matrix, ones_matrix)

        return seq_x, seq_x_mra, var_sp_matrix

    def __len__(self):
        if self.set_type == 0:
            return (len(self.data_x) - self.input_len - self.pred_len + 1) // self.batch_size
        else:
            return len(self.data_x) - self.input_len - self.pred_len + 1

    def train_shuffle(self):
        if self.set_type == 0:
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
