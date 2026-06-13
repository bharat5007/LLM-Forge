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


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        output = self.layer1(x)
        output = output * torch.sigmoid(output)
        return output * self.layer2(x)
