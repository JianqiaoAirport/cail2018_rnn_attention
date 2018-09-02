import json
import random

#  注意中间的 cail2018_# 要改
target_path = "/home/hangyu/"+"cail2018_2"+"/sources/new1/"

f_reader_merged_data = open("merged_data/merged_data.json",'r',encoding='utf-8')
f_reader_merged_data_jieba = open("merged_data/merged_data_jieba.txt", 'r',encoding='utf-8')

f_reader_er_shen = open("merged_data/er_shen.json",'r',encoding='utf-8')
f_reader_er_shen_jieba = open("merged_data/er_shen_jieba.txt",'r',encoding='utf-8')

f_writer_train = open(target_path+"trainnew1.json",'ab')
f_writer_train_jieba = open(target_path+"trainnew1jieba.txt",'a',encoding='utf-8')

f_writer_valid = open(target_path+"validnew1.json",'ab')
f_writer_valid_jieba = open(target_path+"validnew1jieba.txt",'a',encoding='utf-8')


total_num = 1606855
valid_num = 210000
valid_list = random.sample(range(total_num-1), valid_num)


line = f_reader_merged_data.readline()
line_jieba = f_reader_merged_data_jieba.readline()

count = 0

while line:
    data = json.loads(line.replace('\n', ''))
    strout = json.dumps(data, ensure_ascii=False).encode(encoding='utf-8')
    if count not in valid_list:
        f_writer_train.write(strout + b'\n')
        f_writer_train_jieba.write(line_jieba)
    else:
        f_writer_valid.write(strout + b'\n')
        f_writer_valid_jieba.write(line_jieba)

    line = f_reader_merged_data.readline()
    line_jieba = f_reader_merged_data_jieba.readline()
    count += 1

er_shen_line = f_reader_er_shen.readline()
er_shen_line_jieba = f_reader_er_shen_jieba.readline()

while er_shen_line:
    count += 1
    data = json.loads(er_shen_line.replace('\n', ''))
    strout = json.dumps(data, ensure_ascii=False).encode(encoding='utf-8')
    f_writer_valid.write(strout + b'\n')
    f_writer_valid_jieba.write(er_shen_line_jieba)
    er_shen_line = f_reader_er_shen.readline()
    er_shen_line_jieba = f_reader_er_shen_jieba.readline()

f_reader_merged_data.close()
f_reader_merged_data_jieba.close()

f_reader_er_shen.close()
f_reader_er_shen_jieba.close()

f_writer_train.close()
f_writer_train_jieba.close()

f_writer_valid.close()
f_writer_valid_jieba.close()

import os

os.system("cp "+target_path+"trainnew1.json "+target_path+"aug/trainnew1.json")
os.system("cp "+target_path+"trainnew1jieba.txt "+target_path+"aug/trainnew1jieba.txt")
