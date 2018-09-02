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

def get_score(score):
    predict_labels_list = list()  # 所有的预测结果
    for predictlabel in score:
        xitem = np.argwhere(predictlabel > -0.74).flatten()
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

if __name__=='__main__':
    weights = [ 1.14925,  1.14925, -0.85075,  1.14925,  1.14925,  1.14925]
    sum_scores = np.zeros((len(marked_labels_list), 202), dtype=float)
    for i in range(len(weights)):
        scores_name = scores_names[i]
        score = np.vstack(np.load(scores_path + scores_name))
        sum_scores = sum_scores + score* weights[i]
    f1 = 0.891793
    print('local  f1=%g;' % (f1))
    time0 = time.time()
    starttime = time0
    last_f1 = f1
    best_f1 = f1
    lr = 0.15
    # 更新权重
    f1_list = list()
    w_list = list()
    decay1 = 0.995
    decay2 = 0.95
    decay = decay1
    for i in range(200):
        if i == 50:
            decay = decay2    # 增加下降速度
        lr = lr * decay
        weights = np.asarray(weights)
        update_w = np.zeros(weights.shape)
        print('=='*10, i, lr)
        logging.info("==================== %d %g"%(i,lr))
        print('LAST_F1=', last_f1)
        logging.info('LAST_F1= %g'%last_f1)
        for i in range(len(weights)):
            scores_name = scores_names[i]
            update_w[i] = get_update_weight(scores_name, sum_scores)
        update_w = np.asarray(update_w)
        print('update_w=', update_w)
        logging.info('update_w= %s'% str(update_w))
        weights =  weights + update_w              # 更新
        print('new_w=', weights)
        logging.info('new_w= %s'%(str(weights)))
        for i in range(len(weights)):       # 新的权重组合
            scores_name = scores_names[i]
            score = np.vstack(np.load(scores_path + scores_name))
            sum_scores = sum_scores + score*weights[i]     # 新的 sum_scores
        f1 = get_score(sum_scores)
        print('NEW_F1=',f1)
        logging.info('NEW_F1= %g'%f1)
        if f1 > best_f1:
            best_f1 = f1
            np.save('best_weights.npy', weights)
        f1_list.append(f1)
        w_list.append(weights)
        last_f1 = f1 # 更新 f1
        print('**Best_f1=%f; Speed: %g s / epoch.' % (best_f1, time.time() - time0))
        logging.info('**Best_f1=%f; Speed: %g s / epoch.' % (best_f1, time.time() - time0))
        time0 = time.time()
    print('finished , speed: %g s '%(time.time() - starttime))

def get_score_threshold(score):
    predict_labels_list = list()  # 所有的预测结果
    lastscore = 0.877486
    lastthreshold = 0
    thresholdlist = list(np.arange(-2, 2, 0.02))
    #     print(len(thresholdlist))
    #     print(thresholdlist)
    for threshold in thresholdlist:
        time1 = time.time()
        predict_labels_list = list()  # 所有的预测结果
        for predictlabel in score:
            xitem = np.argwhere(predictlabel > threshold).flatten()
            xitem = sorted(list(xitem))
            predict_new = []
            if (len(xitem) > 0):
                predict_new.append(list(xitem))
            else:
                predict_new.append(list(predictlabel.argsort()[-1:-2:-1]))
            predict_labels_list.extend(predict_new)
        predict_label_and_marked_label_list = zip(predict_labels_list, marked_labels_list)
        f1 = judger.get_taskaccu_score(predict_label_and_marked_label_list)
        print('阈值：%.4f' % threshold)
        print('分数：%.6f' % f1)
        if f1 > lastscore:
            lastscore = f1
            lastthreshold = threshold
    print('最优阈值：%.4f' % lastthreshold)
    print('最优分数：%.6f' % lastscore)
    print(time.time() - time0)
    #         break
    # sys.exit()

