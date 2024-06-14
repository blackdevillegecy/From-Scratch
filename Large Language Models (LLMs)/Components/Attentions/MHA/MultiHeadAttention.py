import numpy as np
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, q_dim: int, k_dim: int, v_dim: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.w_query = nn.Parameter(torch.randn(q_dim, embed_dim))
        self.w_key = nn.Parameter(torch.randn(k_dim, embed_dim))
        self.w_value = nn.Parameter(torch.randn(v_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.w_query.matmul(x)
        key = self.w_key.matmul(x)
        value = self.w_value.matmul(x)

        