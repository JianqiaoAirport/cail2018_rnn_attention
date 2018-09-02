# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pickle
import sys
import os

import config

# sys.path.append('../')
# from .data_helpers import pad_X30
# from .data_helpers import pad_X150
# from .data_helpers import pad_X52
# from .data_helpers import pad_X300
# from .data_helpers import train_batch
from data_helpers import eval_batch
from data_helpers import train_batch

""" 把所有的数据按照 batch_size(config.BATCH_SIZE) 进行打包。取 10万 样本作为验证集。
word_title_len = 30.
word_content_len = 150.
char_title_len = 52.
char_content_len = 300.
"""


jieba_train_path = '../'+config.DATA_PATH+'/jieba-data/data_train/'
jieba_valid_path = '../'+config.DATA_PATH+'/jieba-data/data_valid/'
jieba_test_path = '../'+config.DATA_PATH+'/jieba-data/data_test/'
thulac_train_path = '../'+config.DATA_PATH+'/thulac-data/data_train/'
thulac_valid_path = '../'+config.DATA_PATH+'/thulac-data/data_valid/'
thulac_test_path = '../'+config.DATA_PATH+'/thulac-data/data_test/'
paths = [jieba_train_path, jieba_valid_path, jieba_test_path,
         thulac_train_path, thulac_valid_path, thulac_test_path]
for each in paths:
    if not os.path.exists(each):
        os.makedirs(each)

# jieba 数据打包batch
def jieba_train_get_batch( batch_size=config.BATCH_SIZE):
    batch_path = jieba_train_path+'batch/'
    print('loading word train_title and train_content.')
    X_train = np.load(jieba_train_path+'train_data.npy')
    # 训练集打batch
    y_train_law = np.load(jieba_train_path+'train_law_label.npy')
    sample_num = len(X_train)
    print('train_sample_num=%d' % sample_num)
    sample_num = len(y_train_law)
    print('train_sample_num_law=%d' % sample_num)
    train_batch(X_train, y_train_law, batch_path+'law/', batch_size)
    del y_train_law
     
    y_train_accu =  np.load(jieba_train_path+'train_accu_label.npy')
    sample_num = len(y_train_accu)
    print('train_sample_num_accu=%d' % sample_num)
    train_batch(X_train, y_train_accu, batch_path+'accu/', batch_size)
    del y_train_accu
     
    y_train_time =  np.load(jieba_train_path+'train_time_label.npy')
    sample_num = len(y_train_time)
    print('train_sample_num_time=%d' % sample_num)
    train_batch(X_train, y_train_time, batch_path+'time/', batch_size)
    del y_train_time
     
    y_train_timelog = np.load(jieba_train_path+'train_time_labellog.npy')
    sample_num = len(y_train_timelog)
    print('train_sample_num_timelog=%d' % sample_num)
    train_batch(X_train, y_train_timelog, batch_path+'timelog/', batch_size)
    del y_train_timelog,X_train
    
    batch_path = jieba_valid_path+'batch/'
    X_valid = np.load(jieba_valid_path+'valid_data.npy')
    # 验证集打batch
    sample_num = len(X_valid)
    print('valid_sample_num=%d' % sample_num)
    y_valid_law = np.load(jieba_valid_path+'valid_law_label.npy')
    sample_num = len(y_valid_law)
    print('valid_sample_num_law=%d' % sample_num)
    train_batch(X_valid, y_valid_law, batch_path+'law/', batch_size)
    del y_valid_law
    
    y_valid_accu = np.load(jieba_valid_path+'valid_accu_label.npy')
    sample_num = len(y_valid_accu)
    print('valid_sample_num_accu=%d' % sample_num)
    train_batch(X_valid, y_valid_accu, batch_path+'accu/', batch_size)
    del y_valid_accu
    
    y_valid_time = np.load(jieba_valid_path+'valid_time_label.npy')
    sample_num = len(y_valid_time)
    print('valid_sample_num_time=%d' % sample_num)
    train_batch(X_valid, y_valid_time, batch_path+'time/', batch_size)
    del y_valid_time
    
    y_valid_timelog = np.load(jieba_valid_path+'valid_time_labellog.npy')
    sample_num = len(y_valid_timelog)
    print('valid_sample_num_timelog=%d' % sample_num)
    train_batch(X_valid, y_valid_timelog, batch_path+'timelog/', batch_size)
    del y_valid_timelog,X_valid


def jieba_test_get_batch( batch_size=config.BATCH_SIZE):
    test_path = jieba_test_path+'batch/'
    X = np.load(jieba_test_path+'test_data.npy')
    sample_num = len(X)
    print('eval_sample_num=%d' % sample_num)
    eval_batch(X, test_path, batch_size)

# thulac 数据打包batch
def thulac_train_get_batch( batch_size=config.BATCH_SIZE):
    batch_path = thulac_train_path+'batch/'
    print('loading word train_title and train_content.')
    X_train = np.load(thulac_train_path+'train_data_thulac.npy')
    # 训练集打batch
    y_train_law = np.load(thulac_train_path+'train_law_label.npy')
    sample_num = len(X_train)
    print('train_sample_num=%d' % sample_num)
    sample_num = len(y_train_law)
    print('train_sample_num_law=%d' % sample_num)
    train_batch(X_train, y_train_law, batch_path+'law/', batch_size)
    del y_train_law
     
    y_train_accu =  np.load(thulac_train_path+'train_accu_label.npy')
    sample_num = len(y_train_accu)
    print('train_sample_num_accu=%d' % sample_num)
    train_batch(X_train, y_train_accu, batch_path+'accu/', batch_size)
    del y_train_accu
     
    y_train_time =  np.load(thulac_train_path+'train_time_label.npy')
    sample_num = len(y_train_time)
    print('train_sample_num_time=%d' % sample_num)
    train_batch(X_train, y_train_time, batch_path+'time/', batch_size)
    del y_train_time
     
    y_train_timelog = np.load(thulac_train_path+'train_time_labellog.npy')
    sample_num = len(y_train_timelog)
    print('train_sample_num_timelog=%d' % sample_num)
    train_batch(X_train, y_train_timelog, batch_path+'timelog/', batch_size)
    del y_train_timelog,X_train
    
    batch_path = thulac_valid_path+'batch/'
    X_valid = np.load(thulac_valid_path+'valid_data_thulac.npy')
    # 验证集打batch
    sample_num = len(X_valid)
    print('valid_sample_num=%d' % sample_num)
    y_valid_law = np.load(thulac_valid_path+'valid_law_label.npy')
    sample_num = len(y_valid_law)
    print('valid_sample_num_law=%d' % sample_num)
    train_batch(X_valid, y_valid_law, batch_path+'law/', batch_size)
    del y_valid_law
    
    y_valid_accu = np.load(thulac_valid_path+'valid_accu_label.npy')
    sample_num = len(y_valid_accu)
    print('valid_sample_num_accu=%d' % sample_num)
    train_batch(X_valid, y_valid_accu, batch_path+'accu/', batch_size)
    del y_valid_accu
    
    y_valid_time = np.load(thulac_valid_path+'valid_time_label.npy')
    sample_num = len(y_valid_time)
    print('valid_sample_num_time=%d' % sample_num)
    train_batch(X_valid, y_valid_time, batch_path+'time/', batch_size)
    del y_valid_time
    
    y_valid_timelog = np.load(thulac_valid_path+'valid_time_labellog.npy')
    sample_num = len(y_valid_timelog)
    print('valid_sample_num_timelog=%d' % sample_num)
    train_batch(X_valid, y_valid_timelog, batch_path+'timelog/', batch_size)
    del y_valid_timelog,X_valid


def thulac_test_get_batch( batch_size=config.BATCH_SIZE):
    test_path = thulac_test_path+'batch/'
    X = np.load(thulac_test_path+'test_data_thulac.npy')
    sample_num = len(X)
    print('eval_sample_num=%d' % sample_num)
    eval_batch(X, test_path, batch_size)



if __name__ == '__main__':
    # thulac_train_get_batch()
    # thulac_test_get_batch()
    jieba_test_get_batch()
    jieba_train_get_batch()
