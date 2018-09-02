# -*- coding:utf-8 -*-
import numpy as np
import config

def get_batch(data_path, batch_id):
    """get a batch from data_path"""
    new_batch = np.load(data_path + str(batch_id) + '.npz')
    X_batch = new_batch['X']
    y_batch = new_batch['y']
    return [X_batch, y_batch]
import logging
import time
import sys
import os
from judger import Judger
if __name__ == '__main__':
    accusation_path = '../cail_0518/accu.txt'
    law_path = '../cail_0518/law.txt'
    judger = Judger(accusation_path,law_path)
    marked_labels_list = list()
    a = []
    strtime = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    log_name = "../logs/"+strtime+".txt"
    logging.basicConfig(handlers=[logging.FileHandler(log_name, 'w+', 'utf-8')], format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    batchpath = '../data_old/predictbatch/accu/'
    tr_batches = os.listdir(batchpath)  # batch 文件名列表
    n_tr_batches = len(tr_batches)
    X = []
    y = []
    maxitem = 0
    allindex = 0
    count = 0
    threshold = []
    time0 = time.time()
    for batch in range(n_tr_batches):
        [X_batch,  y_batch] = get_batch(batchpath, batch)
        index = 0
        marked_labels_list.extend(y_batch)
        for predictlabel in X_batch:
            X.append(predictlabel)
    lastscore = 0.0
    lastthreshold = 0.0
    thresholdlist = list(np.arange(0.88,1,0.01))
#     print(len(thresholdlist))
#     print(thresholdlist)
    for threshold in thresholdlist:
        time1 = time.time()
        predict_labels_list = list()  # 所有的预测结果
        if config.LAST_LAYER == "sigmoid":
            for predictlabel in X:
                xitem = np.argwhere(predictlabel > threshold).flatten()
                xitem = sorted(list(xitem))
                predict_new = []
                if (len(xitem) > 0):
                    predict_new.append(list(xitem))
                else:
                    predict_new.append(list(predictlabel.argsort()[-1:-2:-1]))
                predict_labels_list.extend(predict_new)
        if config.LAST_LAYER == "softmax":
            for predictlabel in X:
                predict_new = []
                prob = np.argsort(predictlabel)
                pred = [prob[-1]]
                total_prob = config.SOFTMAX_THRESHHOLD - prob[-1]
                for i in range(len(prob))[2:]:
                    total_prob -= prob[-i]
                    if total_prob > 0:
                        pred.append(prob[i - 1])
                    else:
                        break
                xitem = np.array(pred)
                predict_new.append(list(xitem))
                predict_labels_list.extend(predict_new)



        predict_label_and_marked_label_list = zip(predict_labels_list, marked_labels_list)
        score = judger.get_taskaccu_score(predict_label_and_marked_label_list)
        print('阈值：%.4f'%threshold)
        print('分数：%.6f'%score)
        print(time.time() - time1)
        if score>lastscore:
            lastscore = score
            lastthreshold = threshold
    print('最优阈值：%.4f'%lastthreshold)
    print('最优分数：%.6f'%lastscore)
    print(time.time() - time0)
#         break
    sys.exit()