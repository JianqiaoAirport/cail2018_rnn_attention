import numpy as np

a = np.zeros((3,4))

a[1,2:] = 1
a[2, 3] = 1
a[0] = 1



print(a.sum(axis=1))
b = a.sum(axis=1)
b = b[np.newaxis, :]
b = np.transpose(b)

print(b)
# a/a.sum(axis=1)
print(a/b)

print(a)

c = np.zeros(4)
c[3] = 1
c[1] = 4
xitem = np.argwhere(c >= 1).flatten()

print(xitem)

print(type(c.argsort()[-1:-2:-1]))

import json

f_reader = open("test.json",'r',encoding='utf-8')
# f_reader_jieba = open("jieba.txt", 'r',encoding='utf-8')
wf1 = open("test_output.json",'ab')
wf4 = open("trainnew2jieba.txt",'a',encoding='utf-8')

line = f_reader.readline()
while line:

    print(line)

    data = json.loads(line.replace('\n', ''))
    strout = json.dumps(data, ensure_ascii=False).encode(encoding='utf-8')
    wf1.write(strout + b'\n')

    # wf4.write()


    line = f_reader.readline()

import random

print (random.sample(range(20000), 210))

import os

os.system("cp test.json test1.json")


import tensorflow as tf



with tf.Session() as sess:
    a = tf.get_variable(name="a", shape=[1], initializer=tf.constant_initializer(np.array([4.2])))
    b = tf.get_variable(name="b", shape=[2], initializer=tf.constant_initializer(np.array([2])))

    saver = tf.train.Saver()

    saver.restore(sess, "test-2")

    init = tf.global_variables_initializer()

    sess.run(init)

    print(sess.run([a]))


