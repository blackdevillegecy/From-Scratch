import numpy as np
import torch
from torch import nn
from Components.Activations.Softmax import Softmax

class SelfAttention(nn.Module):
    def __init__(self, input_dim: int) -> None:
        self.input_dim = input_dim
        self.w_query = nn.Linear(input_dim, input_dim)
        self.w_key = nn.Linear(input_dim, input_dim)
        self.w_value = nn.Linear(input_dim, input_dim)
        self.softmax = Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.w_query(x)
        key = self.w_key(x)
        value = self.w_value(x)
        
        