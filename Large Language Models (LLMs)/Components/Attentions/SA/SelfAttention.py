import numpy as np
import torch
from torch import nn
from Softmax import Softmax
from BMM import BMM

class SelfAttention(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.w_query = nn.Linear(input_dim, input_dim)
        self.w_key = nn.Linear(input_dim, input_dim)
        self.w_value = nn.Linear(input_dim, input_dim)
        self.softmax = Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.w_query(x)
        key = self.w_key(x)
        value = self.w_value(x)
        obj1 = BMM(query, key.transpose(1, 2))
        scores=obj1.forward()
        scores = scores/(self.input_dim**2)
        attention = self.softmax(scores)
        obj2 = BMM(attention, value)
        weighted = obj2.forward()
        return weighted

if __name__=='__main__':
    x = torch.randn((3, 5, 10))
    input_dim = 10
    obj = SelfAttention(input_dim)
    out = obj.forward(x)
    print(out.shape)