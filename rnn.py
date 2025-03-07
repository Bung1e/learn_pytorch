import torch.nn as nn
import torch

rnn = nn.RNN(33, 64, batch_first=True)

x = torch.randn(8, 3, 33)
h0 = torch.randn(1, 8, 64)
y, h = rnn(x, h0)

print('y:', y.size())
print('h:', h.size())