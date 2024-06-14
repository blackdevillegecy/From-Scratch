import torch
from torch import nn
import numpy as np

class FeedForwardSwiGLU(nn.Module):
    def __init__(self, dim: int, multiplier: int) -> None: 
        super(FeedForwardSwiGLU, self).__init__()
        hidden = multiplier * ((dim + multiplier - 1) // multiplier)

        self.lin1 = nn.Linear(dim, hidden, bias=False)
        self.lin2 = nn.Linear(dim, hidden, bias=False)
        self.lin3 = nn.Linear(hidden, dim, bias=False)

    def _swish(self, x: torch.Tensor, beta: int = 1) -> torch.Tensor:
        return torch.from_numpy(x/(1+np.exp(-beta*x)))
    
    def forward(self, x: torch.Tensor, beta: int = 1) -> torch.Tensor:
        xW_b = self.lin1(x)
        swish_xW_b = self._swish(xW_b.detach().numpy(), beta)
        xV_c = self.lin2(x)
        x = swish_xW_b * xV_c
        x = self.lin3(x)
        return x



## EXPERIMENTS ##
if __name__ == '__main__':
    dim=4
    multiplier=256
    obj = FeedForwardSwiGLU(dim, multiplier)
    x = torch.randn(2, dim)
    sw = obj.forward(x)
    print(sw)