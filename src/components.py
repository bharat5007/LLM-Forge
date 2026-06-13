import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(emb_size))
        self.emb_size = emb_size

    def forward(self, x: torch.Tensor, upsilon: int = 1e-10):
        B, T, C = x.shape
        rms = x.square().mean(-1).sqrt().view(B, T, -1)
        return (x / (rms + upsilon)) * self.gamma
