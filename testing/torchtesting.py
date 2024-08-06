import torch
import torch.nn as nn

A = 9
L = 4

avs = torch.arange(A+1, dtype=torch.long) # (A+1)
print(avs.size())
print(avs)
avs = avs.repeat(L)[L-1:].reshape(1, -1) # (1, A*L+1)
print(avs.size())
print(avs)