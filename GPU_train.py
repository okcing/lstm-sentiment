#coding:utf8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from GPU_model import lstm_sentiment
import numpy as np
import sklearn.metrics as metrics
import random


import pandas as pd #导入Pandas
import jieba
import pickle
import time
start = time.time()

neg=pd.read_excel('data/646864264/neg.xls',header=None,index=None)
pos=pd.read_excel('data/646864264/pos.xls',header=None,index=None)
pos['mark']=1
neg['mark']=0 #给训练语料贴上标签
pn=pd.concat([pos,neg],ignore_index=True) #合并语料
neglen=len(neg)
poslen=len(pos) #计算语料数目

cw = lambda x: list(jieba.cut(x)) #定义分词函数
pn['words'] = pn[0].apply(cw)

word_set = set()
for item in pn['words']:
    for word in item:
        word_set.add(word)

word_to_idx={word:i for i, word in enumerate(word_set)}

print('length', len(word_set))

data = []
for item in pn['words']:
    sent=[]
    for word in item:
        sent.append(word_to_idx[word])
    data.append(sent)
label = []
for item in pn['mark']:
    label.append(item)

#训练数据占三分之一，测试数据占三分之二
data_train = [d for i,d in enumerate(data) if i%3!=0]
label_train = [d for i,d in enumerate(data) if i%3!=0]
data_test = [d for i,d in enumerate(data) if i%3==0]
label_test = [d for i,d in enumerate(data) if i%3==0]



#参数
batch_size = 1
num_layers = 1
num_directions = 1
embedding_size = 300
hidden_size = 256
words_length = len(word_set)
learning_rate = 0.001
num_epochs = 3

lstm = lstm_sentiment(words_length, embedding_size, hidden_size, num_layers, num_directions, batch_size)
lstm.cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(lstm.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    c = list(zip(data_train, label_train))
    random.shuffle(c)
    data_train = [i[0] for i in c]
    label_train = [i[1] for i in c]
    count=0
    for d,l in zip(data_train, label_train):
        count+=1
        if count%100==0:
            print(epoch,count,loss)
        sent = Variable(torch.LongTensor(d)).cuda()
        target = Variable(torch.FloatTensor([int(l)])).cuda()

        optimizer.zero_grad()
        output = lstm(sent)
        output = torch.squeeze(output)

        loss = criterion(output,target)
        loss.backward()
        optimizer.step()

    pre=[]
    for d,l in zip(data_test, label_test):
        sent = Variable(torch.LongTensor(d)).cuda()
        target = Variable(torch.FloatTensor([int(l)]))

        out = lstm(sent)
        if out.cpu().data.numpy()>0.5:
            pre.append(1)
        else:
            pre.append(0)
    #print('target', target, 'out', out)

    acc = metrics.accuracy_score(label_test, pre)
    print(acc)
    end = time.time()
    print(end-start)
    print(pre)