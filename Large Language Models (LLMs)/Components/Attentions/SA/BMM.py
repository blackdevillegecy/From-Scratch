import torch
from torch import nn
import numpy as np

class BMM(nn.Module):
    def __init__(self, inp: torch.Tensor, mat2: torch.Tensor) -> None:
        super(BMM, self).__init__()
        if (inp.shape[0] != mat2.shape[0] or inp.shape[2] != mat2.shape[1]):
            raise ValueError("input shape not matching the mat2 shape!!")
        self.inp = inp ## b x n x m
        self.mat2 = mat2 ## b x m x p
        ## output dimension will be b x n x p
        

    def forward(self, ) -> torch.Tensor:
        out = []
        for bi in range(self.inp.shape[0]):
            m1 = self.inp[bi].detach().numpy()
            m2 = self.mat2[bi].detach().numpy()
            out.append(m1 @ m2)
        out = np.array(out)
        
        if not torch.is_tensor(out):
            out = torch.tensor(out)
        return out
    
if __name__=='__main__':
    b, n, m, p = 10, 2, 3, 4
    inp, mat2 = torch.randn((b, n, m)), torch.randn((b, m, p))
    obj = BMM(inp, mat2)
    out = obj.forward()
    print(out)

