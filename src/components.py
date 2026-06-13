import torch
import math
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


class RoPE(nn.Module):
    def __init__(self):
        super().__init__()

    def calculate_rope(self, pos, dim):
        x = []
        for i in range(dim):
            tensor = torch.zeros(dim)
            if i % 2:
                theta = pos * (10000 ** (-(i - 1) / dim))
                tensor[i - 1] = math.sin(theta)
                tensor[i] = math.cos(theta)
            else:
                theta = pos * (10000 ** (-i / dim))
                tensor[i] = math.cos(theta)
                tensor[i + 1] = -math.sin(theta)

            x.append(tensor)

        return torch.stack(x)

    def forward(self, x: torch.Tensor):
        logits = []
        B, T, C = x.shape

        for i in range(T):
            rope = self.calculate_rope(i, C)
            logits.append(x[:, i, :] @ rope)

        return torch.stack(logits, dim=1)
