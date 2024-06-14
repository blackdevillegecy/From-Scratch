import torch
from torch import nn
import numpy as np

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super(Embedding, self).__init__()
        self.w = nn.Parameter(torch.zeros((vocab_size, embed_dim)))
        self.vocab_size = embed_dim
        nn.init.uniform_(self.w, -0.2, 0.2)
        nn.init.normal_(self.w)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(torch.max(x))
        if (torch.max(x) > embed_dim):
            raise IndexError("max value of x is greater than or equal to vocabulary size!! ")

        return self.w[x]
    
## EXPERIMENT ##

vocab_size = 1200
embed_dim = 12
x = torch.randint(0, 12, size=(4, 4))
obj = Embedding(vocab_size, embed_dim)
em = obj.forward(x)
print(em)