#!usr/bin/python 
# -*- coding:utf-8 -*- 

"""
Construct a Data generator.
"""
import numpy as np
# from tqdm import tqdm
import os

import config


class BatchGenerator(object):
    """ Construct a Data generator. The input X, y should be ndarray or list like type.
    
    Example:
        Data_train = BatchGenerator(X=X_train_all, y=y_train_all, shuffle=True)
        Data_test = BatchGenerator(X=X_test_all, y=y_test_all, shuffle=False)
        X = Data_train.X
        y = Data_train.y
        or:
        X_batch, y_batch = Data_train.next_batch(batch_size)
     """ 
    
    def __init__(self, X, y, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._y = self._y[new_index]
                
    @property
    def X(self):
        return self._X
    
    @property
    def y(self):
        return self._y
    
    @property
    def num_examples(self):
        return self._number_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data 
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]
    
    
def to_categorical_accu(label):
    """把所有的label id 转为 0，1形式。
    Args:
        label: n_sample 个 lists, 文书的罪名标签。每个list对应一个文书，label个数不定。
    return:
        y: ndarray, shape=(sample， n_class)， 其中 n_class = 202.
    Example:
     >>> y_batch = to_categorical(label_batch)
     >>> print(y_batch.shape)
     >>> (10, 202)
    """
    n_sample = len(label)
    y = np.zeros(shape=(n_sample, 202), dtype=np.float32)
    for i in range(n_sample):
        topic_index = label[i]
        y[i, topic_index] = 1

    if config.LAST_LAYER == "softmax":
        y_temp = y.sum(axis=1)
        y_temp = y_temp[np.newaxis, :]
        y_temp = np.transpose(y_temp)
        y = y/y_temp

    return y

def to_categorical_law(label):
    """把所有的label id 转为 0，1形式。
    Args:
        label: n_sample 个 lists, 文书的罪名标签。每个list对应一个文书，label个数不定。
    return:
        y: ndarray, shape=(sample， n_class)， 其中 n_class = 202.
    Example:
     >>> y_batch = to_categorical(label_batch)
     >>> print(y_batch.shape)
     >>> (10, 202)
    """
    n_sample = len(label)
    y = np.zeros(shape=(n_sample, 183), dtype=np.float32)
    for i in range(n_sample):
        topic_index = label[i]
        y[i, topic_index] = 1

    if config.LAST_LAYER == "softmax":
        y_temp = y.sum(axis=1)
        y_temp = y_temp[np.newaxis, :]
        y_temp = np.transpose(y_temp)
        y = y/y_temp
    return y



def train_batch(X, y, batch_path, batch_size=config.BATCH_SIZE):
    """对训练集打batch."""
    if not os.path.exists(batch_path):
        os.makedirs(batch_path)
    sample_num = len(X)
    batch_num = 0
    for start in range(0, sample_num, batch_size):
        end = min(start + batch_size, sample_num)
        batch_name = batch_path+ str(batch_num) + '.npz'
        X_batch = X[start:end]
        y_batch = y[start:end]
        np.savez(batch_name, X=X_batch, y=y_batch)
        batch_num += 1
    print('Finished, batch_num=%d' % (batch_num+1))
    
def train_batch_predict(X, y, batch_path, batch_num, batch_size=config.BATCH_SIZE):
    """对训练集打batch."""
    if not os.path.exists(batch_path):
        os.makedirs(batch_path)
    batch_name = batch_path+ str(batch_num) + '.npz'
    X_batch = X
    y_batch = y
    np.savez(batch_name, X=X_batch, y=y_batch)


def eval_batch(X, batch_path, batch_size=config.BATCH_SIZE):
    """对测试数据打batch."""
    if not os.path.exists(batch_path):
        os.makedirs(batch_path)
    sample_num = len(X)
    print('sample_num=%d' % sample_num)
    batch_num = 0
    for start in range(0, sample_num, batch_size):
        end = min(start + batch_size, sample_num)
        batch_name = batch_path + str(batch_num) + '.npy'
        X_batch = X[start:end]
        np.save(batch_name, X_batch)
        batch_num += 1
    print('Finished, batch_num=%d' % (batch_num+1))
    
# new_batch = np.load('../data/predictbatch/accu/0' + '.npz')
# X_batch = new_batch['X']
# y_batch = new_batch['y']
# print((X_batch))
# print(y_batch)