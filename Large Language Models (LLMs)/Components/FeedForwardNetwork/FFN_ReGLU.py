import torch
from torch import nn
import numpy as np

class FeedForwardReGLU(nn.Module):
    def __init__(self, dim: int, multiplier: int) -> None: 
        super(FeedForwardReGLU, self).__init__()
        hidden = multiplier * ((dim + multiplier - 1) // multiplier)

        self.lin1 = nn.Linear(dim, hidden, bias=False)
        self.lin2 = nn.Linear(dim, hidden, bias=False)
        self.lin3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xW_b = self.lin1(x)
        max_0_xW_b = torch.max(torch.zeros(size=(xW_b.shape)), xW_b)
        xV_c = self.lin2(x)
        x = max_0_xW_b * xV_c
        x = self.lin3(x)

        return x

## EXPERIMENTS ##
dim=4
multiplier=256
obj = FeedForwardReGLU(dim, multiplier)
x = torch.randn(2, dim)
re = obj.forward(x)
print(re)