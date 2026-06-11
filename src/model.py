import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .tokenizer import Tokenizer

import sys
import os

sys.path.insert(0, os.path.abspath(".."))

text = ""


tokenizer = Tokenizer.load("../tokenizer.json")
tokens = tokenizer.encode(text)


class Head(nn.Module):
    def __init__(self, x_emb, head_emb, masking_enabled=False):
        super().__init__()
        self.key = nn.Linear(x_emb, head_emb)
        self.query = nn.Linear(x_emb, head_emb)
        self.value = nn.Linear(x_emb, head_emb)
        self.masking_enabled = masking_enabled

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        logits = q @ k.transpose(-2, -1)
        logits = logits / (k.shape[-1] ** 0.5)

        if self.masking_enabled:
            mask = torch.tril(torch.ones(T, T))
            logits = logits.masked_fill(mask == 0, float("-inf"))

        logits = F.softmax(logits, dim=-1)
        logits = logits @ v
        return logits


class MultiHeadAttention(nn.Module):
    def __init__(self, x_emb, heads_num, masking_enabled=False):
        super().__init__()
        self.proj = nn.Linear(x_emb, x_emb)
        self.heads = nn.ModuleList(
            [Head(x_emb, x_emb // heads_num, masking_enabled) for _ in range(heads_num)]
        )

    def forward(self, x, encoder_logits=None):
        logits = torch.cat(
            [head(x, encoder_logits) for i, head in enumerate(self.heads)], dim=-1
        )
        logits = self.proj(logits)
        return logits


class MultiHeadBlock(nn.Module):
    def __init__(self, x_emb, heads_num, masking_enabled=False):
        super().__init__()
        self.heads = MultiHeadAttention(x_emb, heads_num, masking_enabled)
        self.layer_norm = nn.LayerNorm(x_emb)

    def forward(self, tokens, encoder_logits=None):
        logits_norm = self.layer_norm(tokens)
        logits = self.heads(logits_norm, encoder_logits)
        return logits + tokens


class FeedFwdBlock(nn.Module):
    def __init__(self, x_emb):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(x_emb, 4 * x_emb), nn.GELU(), nn.Linear(4 * x_emb, x_emb)
        )
        self.layer_norm = nn.LayerNorm(x_emb)

    def forward(self, tokens):
        logits = self.layer(tokens)
        logits_norm = self.layer_norm(logits)
        return logits_norm + tokens


class DecoderArchitecture(nn.Module):
    def __init__(self, vocab_size, seq_len, x_emb, heads_num, encoder_num):
        super().__init__()
        self.self_attention = MultiHeadBlock(
            x_emb, heads_num, cross_attention=False, masking_enabled=True
        )
        self.cross_attention = MultiHeadBlock(x_emb, heads_num, masking_enabled=False)
        self.feed_fwd = FeedFwdBlock(x_emb)

    def forward(self, x, encoder_hidden):
        logits = self.self_attention(x)
        logits = self.cross_attention(logits, encoder_hidden)
        logits = self.feed_fwd(logits)
        return logits


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_len,
        x_emb,
        heads_num,
        decoder_num: int = 6,
        encoder_num: int = 6,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = None
        self.architecture = nn.ModuleList(
            [
                DecoderArchitecture(vocab_size, seq_len, x_emb, heads_num, encoder_num)
                for _ in range(decoder_num)
            ]
        )
        self.linear = nn.Linear(x_emb, vocab_size)
        self.look_up_table = nn.Parameter(torch.randn((vocab_size + 2, x_emb)))
        self.postional_enc = nn.Parameter(torch.randn((seq_len, x_emb)))

    def forward(self, tokens, target=None):
        encoder_hidden = self.encoder.encode(tokens)
        loss = None
        B, T = tokens.shape
        x = self.look_up_table[tokens] + self.postional_enc[torch.arange(T)]

        for decoder in self.architecture:
            x = decoder(x, encoder_hidden)

        x = self.linear(x)

        if target is not None:
            target = torch.tensor(target)
            B, T, C = x.shape
            loss = F.cross_entropy(x.view(B * T, C), target.view(B * T))

        return x, loss

    def fit(self, tokens, epochs=100, batch_size=32):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        if len(tokens) % (self.seq_len + 1) != 0:
            pad = self.seq_len + 1 - len(tokens) % (self.seq_len + 1)
            tokens.extend([0] * pad)

        tokens = torch.tensor(tokens)
        tokens = tokens.view(-1, self.seq_len + 1)
        x_chunks = tokens[:, :-1]
        y_chunks = tokens[:, 1:]

        for epoch in range(epochs):
            perm = torch.randperm(x_chunks.size(0))
            x = x_chunks[perm]
            y = y_chunks[perm]

            i = random.randint(0, len(x) - 1 - batch_size)
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]

            output, loss = self.forward(x_batch, y_batch)
            loss.backward()
            print(f"epoch: {epoch}, loss: {loss.item():.4f}")
            optimizer.step()
            optimizer.zero_grad()
