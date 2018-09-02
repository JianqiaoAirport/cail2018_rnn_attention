# -*- coding:utf-8 -*- 

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
# import word2vec
import pickle
import os
import gensim
import sys

SPECIAL_SYMBOL = ['<PAD>', '<UNK>']  # add these special symbols to word(char) embeddings.


def get_word_embedding():
    """提取词向量，并保存至 ../data/word_embedding.npy"""
    print('getting the word_embedding.npy')
    word_embedding = np.load('../data/word_embedding.npy')
    print(word_embedding[0])
    print(word_embedding[2])
    print(word_embedding.shape)
    sys.exit()
    model = gensim.models.KeyedVectors.load_word2vec_format('E:\\Lawresearchcup\\data\\word2vec\\pre_word2vecbig.vector',unicode_errors='ignore', encoding='utf-8')
    words = model.vocab
    print(len(words))
    word_embedding = np.zeros((len(words),300))
    index = 0
    for word in words:
        word_embedding[index] = model[word]
        index+=1
    n_special_sym = len(SPECIAL_SYMBOL)
#     print(words)
    sr_id2word = pd.Series(words, index=range(n_special_sym, n_special_sym + len(words)))
    sr_word2id = pd.Series(range(n_special_sym, n_special_sym + len(words)), index=words)
    # 添加特殊符号：<PAD>:0, <UNK>:1
    embedding_size = 300
    vec_special_sym = np.random.randn(n_special_sym, embedding_size)
    for i in range(n_special_sym):
        sr_id2word[i] = SPECIAL_SYMBOL[i]
        sr_word2id[SPECIAL_SYMBOL[i]] = i
    print(word_embedding.shape)
    word_embedding = np.vstack([vec_special_sym, word_embedding])
    # 保存词向量
    save_path = '../data/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + 'word_embedding.npy', word_embedding)
    # 保存词与id的对应关系
    with open(save_path + 'sr_word2id.pkl', 'wb') as outp:
        pickle.dump(sr_id2word, outp)
        pickle.dump(sr_word2id, outp)
    print('Saving the word_embedding.npy to ../data/word_embedding.npy')


def get_thulac_embedding():
    """提取thulac词向量，并保存至 ../data/char_embedding.npy"""
    model = gensim.models.KeyedVectors.load_word2vec_format('E:\\Lawresearchcup\\data\\word2vec\\pre_word2vecthulacbig.vector',unicode_errors='ignore', encoding='utf-8')
    words = model.vocab
    print(len(words))
    word_embedding = np.zeros((len(words),300))
    index = 0
    for word in words:
        word_embedding[index] = model[word]
        index+=1
    n_special_sym = len(SPECIAL_SYMBOL)
#     print(words)
    sr_id2word = pd.Series(words, index=range(n_special_sym, n_special_sym + len(words)))
    sr_word2id = pd.Series(range(n_special_sym, n_special_sym + len(words)), index=words)
    # 添加特殊符号：<PAD>:0, <UNK>:1
    embedding_size = 300
    vec_special_sym = np.random.randn(n_special_sym, embedding_size)
    for i in range(n_special_sym):
        sr_id2word[i] = SPECIAL_SYMBOL[i]
        sr_word2id[SPECIAL_SYMBOL[i]] = i
    print(word_embedding.shape)
    word_embedding = np.vstack([vec_special_sym, word_embedding])
    # 保存词向量
    save_path = '../data/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + 'thulac_embedding.npy', word_embedding)
    print(word_embedding.shape)
    # 保存词与id的对应关系
    with open(save_path + 'sr_thulac2id.pkl', 'wb') as outp:
        pickle.dump(sr_id2word, outp)
        pickle.dump(sr_word2id, outp)
    print('Saving the thulac_embedding.npy to ../data/thulac_embedding.npy')


if __name__ == '__main__':
#     get_word_embedding()
    get_thulac_embedding()
