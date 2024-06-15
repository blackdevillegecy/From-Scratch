import torch
from torch import nn
import numpy as np

class Softmax(nn.Module):
    def __init__(self, dim: int) -> None:
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num = torch.exp(x)
        denom = torch.sum(torch.exp(x), dim=self.dim, keepdim=True)
        return num/denom
    
## EXPERIMENT ##
if __name__=='__main__':
    dim=0
    x = torch.randn(2, 3)
    obj = Softmax(dim=dim)
    ans = obj.forward(x)
    print(ans)