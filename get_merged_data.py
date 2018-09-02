import json

f_reader_train = open("trainnew1.json",'r',encoding='utf-8')
f_reader_train_jieba = open("trainnew1jieba.txt", 'r',encoding='utf-8')

f_reader_valid = open("validnew1.json",'r',encoding='utf-8')
f_reader_valid_jieba = open("validnew1jieba.txt", 'r',encoding='utf-8')

f_writer_merged_data = open("merged_data/merged_data.json",'ab')
f_writer_merged_data_jieba = open("merged_data/merged_data_jieba.txt",'a',encoding='utf-8')


for _ in range(502):
    valid_line = f_reader_valid.readline()
    valied_line_jieba = f_reader_valid_jieba.readline()


valid_line = f_reader_valid.readline()
valied_line_jieba = f_reader_valid_jieba.readline()
count = 0

while valid_line:
    count += 1
    data = json.loads(valid_line.replace('\n', ''))
    strout = json.dumps(data, ensure_ascii=False).encode(encoding='utf-8')
    f_writer_merged_data.write(strout + b'\n')
    f_writer_merged_data_jieba.write(valied_line_jieba)

    valid_line = f_reader_valid.readline()
    valied_line_jieba = f_reader_valid_jieba.readline()

train_line = f_reader_train.readline()
train_line_jieba = f_reader_train_jieba.readline()

while train_line:
    count += 1
    data = json.loads(train_line.replace('\n', ''))
    strout = json.dumps(data, ensure_ascii=False).encode(encoding='utf-8')
    f_writer_merged_data.write(strout + b'\n')
    f_writer_merged_data_jieba.write(train_line_jieba)
    train_line = f_reader_train.readline()
    train_line_jieba = f_reader_train_jieba.readline()


print("count=%d" % count)
print(count)

f_reader_train.close()
f_reader_train_jieba.close()

f_reader_valid.close()
f_reader_valid_jieba.close()

f_writer_merged_data.close()
f_writer_merged_data_jieba.close()





