import json

f_reader_train_jieba = open("trainnew1jieba.txt", 'r',encoding='utf-8')

train_line_jieba = f_reader_train_jieba.readline()
count = 0

while train_line_jieba:
    count += 1

    train_line_jieba = f_reader_train_jieba.readline()

("count=%d" % count)
print(count)

f_reader_train_jieba.close()

