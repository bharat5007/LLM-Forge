import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from config import ModelConfig
from components import RMSNorm, RoPE

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")


class Head(nn.Module):
    def __init__(
        self,
        emb_size: int,
        head_emb: int,
        q_heads: int,
        kv_heads: int,
        seq_len: int,
        masking_enabled: bool = False,
    ):
        super().__init__()
        self.rope = RoPE(seq_len, head_emb, device)
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_emb = head_emb
        self.masking_enabled = masking_enabled
        self.key = nn.Linear(emb_size, head_emb * kv_heads)
        self.query = nn.Linear(emb_size, head_emb * q_heads)
        self.value = nn.Linear(emb_size, head_emb * kv_heads)

    def forward(self, x):
        B, T, C = x.shape
        v = (
            self.value(x).view(B, T, self.kv_heads, self.head_emb).transpose(1, 2)
        )  # (B, kv_heads, T, head_emb)
        k = (
            self.key(x).view(B, T, self.kv_heads, self.head_emb).transpose(1, 2)
        )  # (B, kv_heads, T, head_emb)
        q = (
            self.query(x).view(B, T, self.q_heads, self.head_emb).transpose(1, 2)
        )  # (B, q_heads, T, head_emb)

        k = k.repeat_interleave(self.q_heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.q_heads // self.kv_heads, dim=1)

        q = self.rope(q)
        k = self.rope(k)  # (B, q_heads, T, head_emb)

        logits = q @ k.transpose(-2, -1)
        logits = logits / (k.shape[-1] ** 0.5)

        if self.masking_enabled:
            mask = torch.tril(torch.ones(T, T, device=logits.device))
            logits = logits.masked_fill(mask == 0, float("-inf"))

        logits = F.softmax(logits, dim=-1)
        logits = logits @ v  # (B, q_heads, T, head_emb)
        logits = logits.transpose(1, 2)  # (B, T, q_heads, head_emb)
        return logits.contiguous().view(B, T, C)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        emb_size: int,
        head_emb: int,
        q_heads: int,
        kv_heads: int,
        seq_len: int,
        masking_enabled: bool = False,
    ):
        super().__init__()
        self.proj = nn.Linear(emb_size, emb_size)
        self.head = Head(
            emb_size, head_emb, q_heads, kv_heads, seq_len, masking_enabled
        )

    def forward(self, x):
        logits = self.head(x)
        logits = self.proj(logits)
        return logits


class MultiHeadBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        head_emb: int,
        q_heads: int,
        kv_heads: int,
        seq_len: int,
        masking_enabled: bool = False,
    ):
        super().__init__()
        self.layer_norm = RMSNorm(emb_size, device)
        self.heads = MultiHeadAttention(
            emb_size, head_emb, q_heads, kv_heads, seq_len, masking_enabled
        )

    def forward(self, tokens):
        logits_norm = self.layer_norm(tokens)
        logits = self.heads(logits_norm)
        return logits + tokens  # [32, seq_len, emb_size]


class SwiGLU(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        hidden_dim = 4 * in_dim
        self.gate_proj = nn.Linear(in_dim, hidden_dim)
        self.up_proj = nn.Linear(in_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor):
        gate = self.gate_proj(x)
        output = gate * torch.sigmoid(gate)
        return self.down_proj(output * self.up_proj(x))


class FeedFwdBlock(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.layer_norm = RMSNorm(emb_size, device)
        self.layer = SwiGLU(emb_size)

    def forward(self, tokens):
        logits_norm = self.layer_norm(tokens)
        logits = self.layer(logits_norm)
        return logits + tokens


class DecoderArchitecture(nn.Module):
    def __init__(
        self,
        emb_size: int,
        head_emb: int,
        q_heads: int,
        kv_heads: int,
        seq_len: int,
    ):
        super().__init__()
        self.self_attention = MultiHeadBlock(
            emb_size, head_emb, q_heads, kv_heads, seq_len, True
        )
        self.feed_fwd = FeedFwdBlock(emb_size)

    def forward(self, hidden_states):
        logits = self.self_attention(hidden_states)
        logits = self.feed_fwd(logits)  # logits = [32, seq_len, emb_size]
        return logits  # [32, seq_len, emb_size]


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.architecture = nn.ModuleList(
            [
                DecoderArchitecture(
                    cfg.emb_size, cfg.heads_emb, cfg.q_heads, cfg.kv_heads, cfg.seq_len
                )
                for _ in range(cfg.decoder_num)
            ]
        )
        self.linear = nn.Linear(cfg.emb_size, cfg.vocab_size)
        self.look_up_table = nn.Parameter(
            torch.randn((cfg.vocab_size + 2, cfg.emb_size)).to(device)
        )
        self.postional_enc = nn.Parameter(
            torch.randn((cfg.seq_len, cfg.emb_size)).to(device)
        )
        self.layer_norm = RMSNorm(cfg.emb_size, device)

    def forward(self, tokens, target=None):
        loss = None
        B, T = tokens.shape  # [32, seq_len]

        # Encoding -> tokenizer + postional
        hidden_states = (
            self.look_up_table[tokens]
            + self.postional_enc[torch.arange(T, device=device)]
        )

        for decoder in self.architecture:
            hidden_states = decoder(hidden_states)

        hidden_states = self.layer_norm(hidden_states)  # [32, seq_len, emb_size]
        logits = self.linear(hidden_states)  # [32, seq_len, vocab_size]

        if target is not None:
            target = target.to(device)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), target.view(B * T))

        return logits, loss

    def fit(self, tokens: list, cfg: ModelConfig):
        # tokens = list of numbers, text/document is passed through tokenizer and tokenzier wrods ko break karke hrr new word ko ek number assign karta ha
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

        # we have to make sure len(tokens) % self.seq_len = 0
        if len(tokens) % (cfg.seq_len + 1) != 0:
            pad = cfg.seq_len + 1 - len(tokens) % (cfg.seq_len + 1)
            tokens.extend([50001] * pad)

        # tokens ko torch ma convert kr rhe ha
        tokens = torch.tensor(tokens, device=device)

        # converting it into a matrix of shape [n, seq_len+1]
        tokens = tokens.view(-1, cfg.seq_len + 1)  # [n, seq_len + 1]

        # preparing x and y chunks
        x_chunks = tokens[:, :-1]  # [n, seq_len]
        y_chunks = tokens[:, 1:]  # [n, seq_len]

        for epoch in range(cfg.epochs):
            # will generate a random permutation of length x[0], for eg. [5,2,3,0,1,4]
            perm = torch.randperm(x_chunks.size(0), device=device)

            # mixing up x and y using same permutation
            x = x_chunks[perm]
            y = y_chunks[perm]

            i = random.randint(0, len(x) - 1 - cfg.batch_size)
            x_batch = x[i : i + cfg.batch_size]  # [32, seq_len]
            y_batch = y[i : i + cfg.batch_size]  # [32, seq_len]

            output, loss = self.forward(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"epoch: {epoch + 1}, loss: {loss.item():.4f}")
            if epoch % 100 == 0:
                torch.save(
                    {
                        "model_state": self.state_dict(),
                        "config": cfg,
                    },
                    "checkpoint.pt",
                )
