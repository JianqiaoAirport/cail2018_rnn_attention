# -*- coding:utf-8 -*-
import gensim
import jieba
import json
import random
from random import choice
from hanziconv import HanziConv
import pickle
jiebainit = jieba
def constructedtext(rate = 0.2, replicationtimes = 1):
    print('start constructedtext')
    sourcefile = '../sources/new/trainnew.json'
    addjsonfile = '../cail_0518/aug/addfile.json'
    f = open(sourcefile,'r',encoding = 'utf-8')
    wf = open(addjsonfile,'ab')
    print('load word2vec model')
    model = gensim.models.KeyedVectors.load_word2vec_format('../result/pre_word2vecjiebawordbig.vector',unicode_errors='ignore', encoding='utf-8')
    line = f.readline()
    wordlist = ['被告人','被害人','结论','上诉人','处理','鉴定','事实','驾驶证','笔录','精神','公安局','伤害','客观','检察院','驾驶','医院','中心','指控','刑事','赃款','机动车','人口','审理','交通设施', '贩卖', '狩猎', '传播', '安全', '开发票', '吸毒', '引诱', '系统', '犯罪', '毁灭', '携带', '交易', '军事设施', '珍贵', '传销', '牟利', '伪劣', '制品', '节育手术', '假药', '招收', '管理', '放火', '收购', '商品', '打击报复', '植物', '污染环境', '抢夺', '移交', '活动', '警用', '制毒', '诈骗', '种植', '出口', '无线电', '交通秩序', '贷款', '公共安全', '古墓葬', '收买', '传授', '个人信息', '挪用公款', '性质', '帐簿', '贪污', '间谍', '虚开', '影响力', '扣押', '诬告陷害', '敲诈勒索', '巨额财产', '盗窃', '不征', '杀害', '专用发票', '爆炸物', '经济', '危及', '伪劣产品', '有害', '电力设备', '窃取', '假冒', '禁止', '逃税', '动植物', '采伐', '船只', '组织', '非法组织', '肇事', '发票', '寻衅滋事', '所得', '强奸', '欺骗', '公务员', '动物', '销售', '裁定', '重伤', '拐卖妇女', '企业', '驾驶', '职务侵占', '妨害', '林木', '私分', '滥伐林木', '标准', '单位', '捕捞', '注册资本', '住宅', '赌博', '贿赂', '聚众斗殴', '购买', '破坏', '秩序', '非国家', '进出口', '监管', '设备', '侵占', '冲击', '经营', '印章', '盗掘', '强制', '作证', '高利', '事业单位', '犯', '注册商标', '集资', '变造', '凭证', '买卖', '淫秽物品', '武器弹药', '财产', '赃物', '帮助', '盗伐', '符合', '运输', '工作人员', '处罚', '保险', '遗弃', '虚', '公用', '非法经营', '弹药', '猥亵', '收益', '聚众', '财务会计', '物品', '假币', '危险', '物质', '植物种子', '遗址', '重点保护', '损坏', '存款', '故意伤害', '卖血', '私藏枪支', '掩饰', '通信', '农药', '侮辱', '武装部队', '装备', '串通', '拐卖', '土地', '投标', '居民身份证', '汽车', '社会秩序', '报告', '挪用资金', '标识', '销毁', '水产品', '抵扣', '国有资产', '窝藏', '武器装备', '重婚', '滥用职权', '检疫', '被监管人', '著作权', '倒卖', '违法', '强迫', '采矿', '学生', '会计凭证', '卖淫', '行医', '伪造', '转移', '虚报', '编造', '财物', '黑社会', '幼苗', '工具', '妇女儿童', '进行', '计算机信息', '劳动报酬', '伪证', '绑架', '徇私枉法', '失火', '货币', '易燃易爆', '吸收', '公共场所', '票据', '废物', '提供', '证件', '方法', '致人', '增值税', '侵入', '非法拘禁', '交通肇事', '邮寄', '车票', '领导', '杀人', '有毒', '管制', '玩忽职守', '信用卡', '妇女', '滥伐', '挪用', '人民团体', '使用', '濒危', '劫持', '器材', '刀具', '合同诈骗', '刑事案件', '控制', '被', '协助', '扰乱', '拒不执行', '参加', '秘密', '出售', '种子', '毒赃', '特定', '拐骗', '公文', '农用地', '军人', '少征', '古文化', '化肥', '死亡', '货物', '安全事故', '赌场', '兽药', '用于', '故意', '抢劫', '投放', '复制', '徇私舞弊', '储存', '洗钱', '公务', '公民', '性病', '生产', '设施', '信息', '利用', '广播电视', '来源不明', '交通工具', '他人', '重大', '危险物品', '使用权', '制作', '虚假', '冒充', '恐怖', '隐匿', '侵犯', '隐瞒', '危害', '制造', '获取', '票证', '责任事故', '占用', '电信', '会计', '处置', '招摇撞骗', '介绍', '查封', '退税', '款物', '证人', '普通', '加工', '食品', '公众', '盗伐林木', '对', '行贿', '诽谤', '脱逃', '转让', '教唆', '税款', '过失', '通讯', '转贷', '不', '虐待', '国家机关', '以', '劳动', '受贿', '原', '开设', '专用', '国家', '持有', '出版', '哄抢', '逃避', '承兑', '儿童', '包庇', '枪支弹药', '公司', '船票', '尸体', '金融', '非法', '拒', '军事', '容留', '支付', '犯罪分子', '有价', '发放贷款', '骗取', '猎捕', '冻结', '伪造证据', '野生动物', '走私', '毒品', '文物', '毁坏', '判决', '程序', '爆炸']
    count = 0
    while line:
        data = json.loads(line.replace('\n',''))
        tmp = data
        cuttextlist = jiebainit.cut(data['fact'].replace('\r','').replace('\n','').replace('\t',''))
        cuttextdict = {}
        index = 0
        for i in cuttextlist:
            cuttextdict[index] = i
            index += 1
        length = index*rate
        for _ in range(replicationtimes):
            indexlist = random.sample(cuttextdict.keys(), int(length))
            for i in indexlist:
                if cuttextdict[i] in wordlist:
                    continue
                try:
                    word = choice(model.most_similar(cuttextdict[i])) #随机从最相似的十个单词中选择 一个
#                     word = model.most_similar(cuttextdict[i])
                    cuttextdict[i] = word[0]
                except Exception as err:
                    continue
            content = ''.join(cuttextdict.values())
            tmp['fact'] = content
            strout = json.dumps(tmp, ensure_ascii=False).encode(encoding='utf_8')
            wf.write(strout+b'\n')  
        count += 1    
        line = f.readline()
        if(count > 100):
            break
    return

def addfiletotraindata():
    print('start addfiletotraindata')
    # 加载过滤词
    filter_read = open("../stopwords/stopwords.txt", mode='r', encoding='utf-8')
    filter_words = set()
    for words in filter_read:
        words = words.strip("\n")
        filter_words.add(words)
    filter_read.close()
    
    print('start')
    file1 = '../cail_0518/aug/addfile.json'
    f = open(file1,'r',encoding='utf-8')
    wfile1 = '../sources/new/aug/trainnew.json'
    wf1 = open(wfile1,'ab')
    line = f.readline()
    alltext = []
    index = 0
    while line :
        d = json.loads(line.replace('\n',''))
        strout = json.dumps(d, ensure_ascii=False).encode(encoding='utf_8')
        wf1.write(strout+b'\n')
        alltext.append(d['fact'].replace('\r','').replace('\n','').replace('\t','')) 
        line = f.readline()
        index += 1
    print(index)
    print(len(alltext))
    f.close()
    wf1.close()
    wfile2 = '../sources/new/aug/trainnewjieba.txt'
    wf2 = open(wfile2,'a',encoding = 'utf-8')
    wfile3 = '../cail_0518/aug/addfilejieba.txt'
    wf3 = open(wfile3,'a',encoding = 'utf-8')
    count = 0
    for title in alltext:
        document = HanziConv.toSimplified(title)
        count += 1
        if count % 2000 == 0:
            print(count)
        seg_list = jiebainit.cut(document, cut_all=True)  # seg_list是生成器generator类型
        # 去掉过滤词
        splited_words = []
        for seg in seg_list:
            if seg in filter_words:
                continue
            if len(seg)<1:
                continue
            splited_words.append(seg)
        words = ' '.join(splited_words) 
        words = words.strip() + "\n"
        wf2.write(words)
        wf3.write(words)
    print(count)
    wf2.close()
    wf3.close()
    return index+count

def checkissplitright(maxindex):
    print('start checkissplitright')
    wfile1 = '../sources/new/aug/trainnew.json'
    wfile2 = '../sources/new/aug/trainnewjieba.txt'
    f = open(wfile1,'r',encoding='utf-8')  
    line = f.readline()
    index = 0
    while line :
        d = json.loads(line.replace('\n',''))
        if index in [1,500,8000,maxindex-20,maxindex-3]:
            print(d)
        line = f.readline()
        index += 1
    print(index)
    f.close()   
    f = open(wfile2,'r',encoding='utf-8')  
    line = f.readline()
    index = 0
    while line :
        if index in [1,500,8000,maxindex-20,maxindex-3]:
            print(index)
            print(line.replace('\n',''))
        line = f.readline()
        index += 1
    print(index)
    f.close()   

jieba_train_path = '../data/new/aug/jieba-data/data_train/'
trainsourcefile = '../sources/new/aug/trainnew.json'    
trainjiebafile = '../sources/new/aug/trainnewjieba.txt'
paths = [jieba_train_path]
import os
for each in paths:
    if not os.path.exists(each):
        os.makedirs(each)    
 
def init():
    f = open('../cail_0518/law.txt', 'r', encoding = 'utf8')
    law = {}
    lawname = {}
    line = f.readline()
    while line:
        lawname[len(law)] = line.strip()
        law[line.strip()] = len(law)
        line = f.readline()
    f.close()


    f = open('../cail_0518/accu.txt', 'r', encoding = 'utf8')
    accu = {}
    accuname = {}
    line = f.readline()
    while line:
        accuname[len(accu)] = line.strip()
        accu[line.strip()] = len(accu)
        line = f.readline()
    f.close()


    return law, accu, lawname, accuname


law, accu, lawname, accuname = init()
import numpy as np
def get_labels():
    print('start get_labels')
    global law
    global accu
    # 做单标签
    """获取训练集所有样本的标签。注意之前在处理数据时丢弃了部分没有 title 的样本。"""
    #tain数据集做标签
    f = open(trainsourcefile,mode='r',encoding = 'utf-8')
    line = f.readline()
    index = 0
    law_labellist = []
    accu_labellist = []
#     time_labellist = []
#     time_labellistlog = []
    while line:
        data = json.loads(line.replace('\n',''))
        lawsjson = data['meta']['relevant_articles']
        accusjson = data['meta']['accusation']
#         timesjson = data['meta']['term_of_imprisonment']
        lawslabel = []
        accuslabel = []
        for lawitem in lawsjson:
            lawslabel.append(law[str(lawitem)])
        law_labellist.append(lawslabel)
        for accitem in accusjson:
            accuslabel.append(accu[accitem])
        accu_labellist.append(accuslabel)
#         time_labellist.append(gettimeclass(gettime(timesjson)))
#         time_labellistlog.append(gettime(timesjson, type=True))
        index +=1
        line = f.readline()
    f.close()
    print(index)
    print(len(law_labellist))
    print(len(accu_labellist))
#     print(len(time_labellist))
#     print(len(time_labellistlog))
    np.random.seed(100)
    new_index = np.random.permutation(index)
    law_labellist = np.asarray(law_labellist)[new_index]
    accu_labellist = np.asarray(accu_labellist)[new_index]
    np.save(jieba_train_path+'train_law_label.npy',law_labellist)
    np.save(jieba_train_path+'train_accu_label.npy',accu_labellist)
    del law_labellist, accu_labellist
    return new_index, index

def pad_X400(words, max_len=400):
    """把 jiebatext 整理成固定长度。
    """
    words = list(words)
    words_len = len(words)
    words = np.asarray(words)
    if words_len == max_len:
        return words
    if words_len > max_len:
        return words[-max_len:]
    return np.hstack([words, np.zeros(max_len-words_len, dtype=int)])

with open('../data/sr_word2id.pkl', 'rb') as inp:
        sr_id2word = pickle.load(inp)
        sr_word2id = pickle.load(inp)
dict_word2id = dict()
for i in range(len(sr_word2id)):
    dict_word2id[sr_word2id.index[i]] = sr_word2id.values[i]

def get_id(word):
    
    """获取 word 所对应的 id.
    如果该词不在词典中，用 <UNK>（对应的 ID 为 1 ）进行替换。
    """
    if word not in dict_word2id:
        return 1
    else:
        return dict_word2id[word]
def get_id4words(words):
    """把 words 转为 对应的 id"""
    ids = map(get_id, words)  # 获取id
    return ids

def get_datas_jiebaaug(length):
    print('start get_datas_jiebaaug')
    """转换jieba数据集成数组，shape(,400)"""
    print("转换jieba数据集成数组，shape(,400)")
    f = open(trainjiebafile,mode='r',encoding = 'utf-8')
    line = f.readline()
    index = 0
    dataarray = np.zeros((length,400),dtype = int)
#     dataarray = np.zeros((1416587,400),dtype = int)
    while line:
        data = line.replace('\n','').split(' ')
        a = get_id4words(data)
        dataarray[index] = np.asarray(pad_X400(a))
        index +=1
        line = f.readline()
    f.close()
    print(index)
    np.random.seed(100)
    new_index = np.random.permutation(index)
    datanewarray = dataarray[new_index]
    del dataarray,new_index
    print(datanewarray.shape)
    np.save(jieba_train_path+'train_data.npy', datanewarray)
    del datanewarray
    print('--save train successfully--')
import time
if __name__=='__main__':
    print('start')
    time0 = time.time()
    constructedtext(0.2,2)
    maxindex = addfiletotraindata()
    print(maxindex)
    checkissplitright(maxindex)
    newindex,length = get_labels()
    np.save('../data/new/aug/index.npy',newindex)
    print("cost time %.4f"%(time.time() - time0))
    get_datas_jiebaaug(length)