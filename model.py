import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class lstm_sentiment(nn.Module):

    def __init__(self, words_length, embedding_size, hidden_size, num_layers, num_directions, batch_size):
        super(lstm_sentiment, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(words_length, embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True)
        self.ln1 = nn.Linear(hidden_size, 128)
        self.ln2 = nn.Linear(128, 1)


    #hidden和cell的初始化

    def forward(self, x):
        #x是一句话，x的维度应该是(1, len(sent))
        w2v = self.embedding(x)
        #w2v的维度应该是(len(sent), emnbdeeing_size)
        #输入lstm,w2v里面，batch=1，seq_len长度为一句话的长度，每个单词的维度是embedding的维度
        w2v = w2v.view(1, -1, self.embedding_size)

        #初始化
        h0 = Variable(torch.zeros(self.batch_size, self.num_layers*self.num_directions, self.hidden_size))
        c0 = Variable(torch.zeros(self.batch_size, self.num_layers*self.num_directions, self.hidden_size))
        out, _ = self.lstm(w2v, (h0, c0))

        #这里out应该是hidden_size维的
        out = F.sigmoid(self.ln1(out[:,-1,:]))
        out = F.sigmoid(self.ln2(out))

        return out



