# -*- coding:utf-8 -*-
import numpy as np
from numpy import sort
import json
def dataset_split():
    sourcefile = '../cail_0518/big/cail2018_big_filtered_new.json'
    jiebafile = '../sources/jieba_text/jieba_words_big_filtered_new.txt'
    path = '../sources/new1/'
    wtrainsource = path+'trainnew1.json'
    wvalidsource = path+'validnew1.json'
    wtestsource = path+'testnew1.json'
    wtrainjieba = path+'trainnew1jieba.txt'
    wvalidjieba = path+'validnew1jieba.txt'
    wtestjieba = path+'testnew1jieba.txt'
    fsource = open(sourcefile,'r',encoding='utf-8')
    fjieba = open(jiebafile,'r',encoding = 'utf-8')
    np.random.seed(66)
    indexnew = np.random.permutation(np.arange(1706855))
    np.save('../sources/new1/indexnew.npy',np.asarray(indexnew))
    validnum = 210000
    testnum = 100000
    validindexnew = sort(indexnew[:validnum])
    np.save(path+'npy/validindexnew.npy',np.asarray(validindexnew))
    testindexnew = sort(indexnew[validnum:validnum+testnum])
    np.save(path+'npy/testindexnew.npy',np.asarray(testindexnew))
    trainindexnew = sort(indexnew[validnum+testnum:])
    np.save(path+'npy/trainindexnew.npy',np.asarray(trainindexnew))
    linesource = fsource.readline()
    linejieba = fjieba.readline()
    wf1 = open(wtrainsource,'ab')
    wf2 = open(wvalidsource,'ab')
    wf3 = open(wtestsource,'ab')
    wf4 = open(wtrainjieba,'a',encoding='utf-8')
    wf5 = open(wvalidjieba,'a',encoding='utf-8')
    wf6 = open(wtestjieba,'a',encoding='utf-8')
    index = 0
    while linesource:
        data = json.loads(linesource.replace('\n',''))
        strout = json.dumps(data, ensure_ascii=False).encode(encoding='utf-8')
        if(index in trainindexnew):
            wf1.write(strout+b'\n')
            wf4.write(linejieba)
        if(index in validindexnew):
            wf2.write(strout+b'\n')
            wf5.write(linejieba)
        if(index in testindexnew):
            wf3.write(strout+b'\n')
            wf6.write(linejieba)
        index += 1
        linesource = fsource.readline()
        linejieba = fjieba.readline()
    fsource.close()
    fjieba.close()
    wf1.close()
    wf2.close()
    wf3.close()
    wf4.close()
    wf5.close()
    wf6.close()
    print('--------------------------------')
    path = '../sources/new2/'
    wtrainsource = path+'trainnew2.json'
    wvalidsource = path+'validnew2.json'
    wtestsource = path+'testnew2.json'
    wtrainjieba = path+'trainnew2jieba.txt'
    wvalidjieba = path+'validnew2jieba.txt'
    wtestjieba = path+'testnew2jieba.txt'
    fsource = open(sourcefile,'r',encoding='utf-8')
    fjieba = open(jiebafile,'r',encoding = 'utf-8')
    np.random.seed(33)
    indexnew = np.random.permutation(np.arange(1706855))
    np.save('../sources/new1/indexnew.npy',np.asarray(indexnew))
    validnum = 210000
    testnum = 100000
    validindexnew = sort(indexnew[:validnum])
    np.save(path+'npy/validindexnew.npy',np.asarray(validindexnew))
    testindexnew = sort(indexnew[validnum:validnum+testnum])
    np.save(path+'npy/testindexnew.npy',np.asarray(testindexnew))
    trainindexnew = sort(indexnew[validnum+testnum:])
    np.save(path+'npy/trainindexnew.npy',np.asarray(trainindexnew))
    linesource = fsource.readline()
    linejieba = fjieba.readline()
    wf1 = open(wtrainsource,'ab')
    wf2 = open(wvalidsource,'ab')
    wf3 = open(wtestsource,'ab')
    wf4 = open(wtrainjieba,'a',encoding='utf-8')
    wf5 = open(wvalidjieba,'a',encoding='utf-8')
    wf6 = open(wtestjieba,'a',encoding='utf-8')
    index = 0
    while linesource:
        data = json.loads(linesource.replace('\n',''))
        strout = json.dumps(data, ensure_ascii=False).encode(encoding='utf-8')
        if(index in trainindexnew):
            wf1.write(strout+b'\n')
            wf4.write(linejieba)
        if(index in validindexnew):
            wf2.write(strout+b'\n')
            wf5.write(linejieba)
        if(index in testindexnew):
            wf3.write(strout+b'\n')
            wf6.write(linejieba)
        index += 1
        linesource = fsource.readline()
        linejieba = fjieba.readline()
    fsource.close()
    fjieba.close()
    wf1.close()
    wf2.close()
    wf3.close()
    wf4.close()
    wf5.close()
    wf6.close()
if __name__=='__main__':
    dataset_split()