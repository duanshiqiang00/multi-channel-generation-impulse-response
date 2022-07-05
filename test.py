import torch
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
# torch.backends.cudnn.enabled = False


rnn = nn.LSTM(10, 20,batch_first=True) # 一个单词向量长度为10，隐藏层节点数为20，LSTM有2层
input = torch.randn(5, 3, 10) # 输入数据由3个句子组成，每个句子由5个单词组成，单词向量长度为10
h0 = torch.randn(1, 5, 20) # 2：LSTM层数*方向 3：batch 20： 隐藏层节点数
c0 = torch.randn(1, 5, 20) # 同上
input=input.to(device)
h0 = h0.to(device)
c0=c0.to(device)
print(input.shape, h0.shape, c0.shape)
rnn = rnn.to(device)
print(rnn)

output, (hn, cn) = rnn(input, (h0, c0))

print(output.shape, hn.shape, cn.shape)

import torch


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gru = torch.nn.GRU(128, 64, 1, batch_first=True, bidirectional=True)

    def forward(self, inp):
        self.gru.flatten_parameters()
        out, _ = self.gru(inp)
        return out


net = Net()
inp = torch.rand(32, 8, 128, requires_grad=True)

net = torch.nn.DataParallel(net)

inp = inp.cuda()
net = net.cuda()
out = net.forward(inp)