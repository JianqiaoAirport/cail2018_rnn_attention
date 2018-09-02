# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import pickle
import os
import sys
import time
import logging
# sys.path.append('..')
from judger import Judger

strtime = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
log_name = "../"+strtime+"ensemble.txt"
logging.basicConfig(handlers=[logging.FileHandler(log_name, 'w+', 'utf-8')], format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

accusation_path = '../cail_0518/accu.txt'
law_path = '../cail_0518/law.txt'
judger = Judger(accusation_path,law_path)

marked_labels_list = np.load('../data/predictbatch/old/accu/accu_labels.npy')
scores_path = '../data/predictbatch/old/accu/'

scores_names = ['accu_bigru.npy','accu_cnn.npy','accu_fasttext.npy','accu_lstm.npy',
               'accu_rcnn128.npy','accu_rcnn256.npy']
scores_name_num = len(scores_names)

def sigmoid_(inputs):

    """

    Calculate the sigmoid for the give inputs (array)

    :param inputs:

    :return:

    """

    sigmoid_scores = [1 / float(1 + np.exp(- x)) for x in inputs]

    return sigmoid_scores

def sigmoid(scores):
    sigmoid_scores = []
    for score in scores:
        sigmoid_scores.append(sigmoid_(score))
    return np.asarray(sigmoid_scores)


def get_score_weight(score, threshold = -0.8):
    predict_labels_list = list()  # 所有的预测结果
    for predictlabel in score:
        xitem = np.argwhere(predictlabel > threshold).flatten()
        xitem = sorted(list(xitem))
        predict_new = []
        if(len(xitem)>0):
            predict_new.append(list(xitem))
        else :
            predict_new.append(list(predictlabel.argsort()[-1:-2:-1]))
        predict_labels_list.extend(predict_new)
    predict_label_and_marked_label_list = zip(predict_labels_list, marked_labels_list)
    f1 = judger.get_taskaccu_score(predict_label_and_marked_label_list)
    return f1

def get_update_weight(score_name, sum_scores):
    """根据线下验证集的 f1 值变化趋势来调整模型的融合权重。
    Args:
        score_name: 需要调整的模型。
    Returns:
        lr: 模型的权重变化。
    """
    global lr   # 权重调整率
    score = np.vstack(np.load(scores_path + score_name))
    new_score = sum_scores + score*lr
    f1 = get_score(new_score)
    if f1 > last_f1:
        return lr
    else:
        new_score = sum_scores - score*lr
        f1 = get_score(new_score)
        if f1 > last_f1:
            return -lr
    return 0.0




