import json

f_reader_valid = open("validnew1.json",'r',encoding='utf-8')
f_reader_jieba = open("validnew1jieba.txt", 'r',encoding='utf-8')

f_writer_er_shen = open("merged_data/er_shen.json",'ab')
f_writer_er_shen_jieba = open("merged_data/er_shen_jieba.txt",'a',encoding='utf-8')



for _ in range(502):
    line = f_reader_valid.readline()
    line_jieba = f_reader_jieba.readline()
    data = json.loads(line.replace('\n', ''))
    strout = json.dumps(data, ensure_ascii=False).encode(encoding='utf-8')
    f_writer_er_shen.write(strout + b'\n')
    f_writer_er_shen_jieba.write(line_jieba)

f_reader_valid.close()
f_reader_jieba.close()

f_writer_er_shen.close()
f_writer_er_shen_jieba.close()

