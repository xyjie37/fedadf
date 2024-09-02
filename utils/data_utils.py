import numpy as np
import os
import torch
from torch.utils.data import DataLoader


def read_data(dataset, idx, task_num=0, is_train=True):
    if is_train:
        train_data_dir = os.path.join('' + dataset, 'train/')

        train_file = train_data_dir + 'client-' + str(idx) + '-task-' + str(task_num) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('' + dataset, 'test/')

        test_file = test_data_dir + 'client-' + str(idx) + '-task-' + str(task_num) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()
        return test_data
    
def read_all_test_data(dataset):
    test_data_dir = os.path.join('' + dataset, 'test/')

    test_file = test_data_dir + 'test-data.npz'
    with open(test_file, 'rb') as f:
        test_data = np.load(f, allow_pickle=True)['data'].tolist()

    X_test = torch.Tensor(test_data['x']).type(torch.float32)
    y_test = torch.Tensor(test_data['y']).type(torch.int64)
    test_data = [(x, y) for x, y in zip(X_test, y_test)]

    return DataLoader(test_data, 50, drop_last=False, shuffle=True)


def read_client_data(dataset, idx, task, is_train=True):

    if is_train:
        train_data = read_data(dataset, idx, task_num=task, is_train=True)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, task_num=task, is_train=False)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def load_train_data(dataset, id, task, batch_size=None):
        if batch_size == None:
            batch_size = 50
        train_data = read_client_data(dataset, id, task, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)

def load_test_data(dataset, task, id, batch_size=None):
    if batch_size == None:
        batch_size = 16
    test_data = read_client_data(dataset, id, task=task, is_train=False)
    return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)