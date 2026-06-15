import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(emb_size))
        self.emb_size = emb_size

    def forward(self, x: torch.Tensor, upsilon: float = 1e-10):
        B, T, C = x.shape
        rms = x.square().mean(-1).sqrt().view(B, T, -1)
        return (x / (rms + upsilon)) * self.gamma


############################# This is discarded due to TC (B*T*C**2) #################################
# class RoPE(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def calculate_rope(self, pos, dim):
#         x = []
#         for i in range(dim):
#             tensor = torch.zeros(dim)
#             if i % 2:
#                 theta = pos * (10000 ** (-(i - 1) / dim))
#                 tensor[i - 1] = math.sin(theta)
#                 tensor[i] = math.cos(theta)
#             else:
#                 theta = pos * (10000 ** (-i / dim))
#                 tensor[i] = math.cos(theta)
#                 tensor[i + 1] = -math.sin(theta)

#             x.append(tensor)

#         return torch.stack(x)

#     def forward(self, x: torch.Tensor):
#         logits = []
#         B, T, C = x.shape

#         for i in range(T):
#             rope = self.calculate_rope(i, C)
#             logits.append(x[:, i, :] @ rope)

#         return torch.stack(logits, dim=1)


class RoPE(nn.Module):
    def __init__(self, T, C):
        super().__init__()
        freqs = self.calculate_freq(T, C)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def calculate_freq(self, T, C):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, C, 2) / C))
        positions = torch.arange(T)
        return torch.outer(positions, inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        out_even = x1 * self.cos - x2 * self.sin
        out_odd = x1 * self.sin + x2 * self.cos
        out = torch.stack((out_even, out_odd), dim=-1)
        out = out.flatten(-2)
        return out
