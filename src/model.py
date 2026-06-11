import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .tokenizer import Tokenizer
from .config import ModelConfig

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
        self.layer_norm = nn.LayerNorm(x_emb)
        self.heads = MultiHeadAttention(x_emb, heads_num, masking_enabled)

    def forward(self, tokens, encoder_logits=None):
        logits_norm = self.layer_norm(tokens)
        logits = self.heads(logits_norm, encoder_logits)
        return logits + tokens


class FeedFwdBlock(nn.Module):
    def __init__(self, x_emb):
        super().__init__()
        self.layer_norm = nn.LayerNorm(x_emb)
        self.layer = nn.Sequential(
            nn.Linear(x_emb, 4 * x_emb), nn.GELU(), nn.Linear(4 * x_emb, x_emb)
        )

    def forward(self, tokens):
        logits_norm = self.layer_norm(tokens)
        logits = self.layer(logits_norm)
        return logits + tokens


class DecoderArchitecture(nn.Module):
    def __init__(self, emb_size, heads_num):
        super().__init__()
        self.self_attention = MultiHeadBlock(
            emb_size, heads_num, masking_enabled=True
        )
        self.feed_fwd = FeedFwdBlock(emb_size)

    def forward(self, hidden_states):
        logits = self.self_attention(hidden_states)
        logits = self.feed_fwd(logits)
        return logits


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.architecture = nn.ModuleList(
            [
                DecoderArchitecture(cfg.emb_size, cfg.heads_num)
                for _ in range(cfg.decoder_num)
            ]
        )
        self.linear = nn.Linear(cfg.emb_size, cfg.vocab_size)
        self.look_up_table = nn.Parameter(torch.randn((cfg.vocab_size + 2, cfg.emb_size)))  # [vocab+2, emb_size]
        self.postional_enc = nn.Parameter(torch.randn((cfg.seq_len, cfg.emb_size)))         # [seq_len, emb_size]
        self.layer_norm = nn.LayerNorm(cfg.emb_size)

    def forward(self, tokens, target=None):
        loss = None
        B, T = tokens.shape                                                                 # [32, seq_len]

        # Encoding -> tokenizer + postional
        hidden_states = self.look_up_table[tokens] + self.postional_enc[torch.arange(T)]    # [32, seq_len, emb_size]

        for decoder in self.architecture:
            hidden_states = decoder(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        logits = self.linear(hidden_states)

        if target is not None:
            target = torch.tensor(target)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), target.view(B * T))

        return logits, loss

    def fit(self, tokens: list, cfg: ModelConfig):
        # tokens = list of numbers, text/document is passed through tokenizer and tokenzier wrods ko break karke hrr new word ko ek number assign karta ha
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

        # we have to make sure len(tokens) % self.seq_len = 0
        if len(tokens) % (cfg.seq_len + 1) != 0:
            pad = cfg.seq_len + 1 - len(tokens) % (cfg.seq_len + 1)
            tokens.extend([0] * pad)

        # tokens ko torch ma convert kr rhe ha
        tokens = torch.tensor(tokens)

        # converting it into a matrix of shape [n, seq_len+1]
        tokens = tokens.view(-1, cfg.seq_len + 1)          # [n, seq_len + 1]

        # preparing x and y chunks
        x_chunks = tokens[:, :-1]                           # [n, seq_len]
        y_chunks = tokens[:, 1:]                            # [n, seq_len]

        for epoch in range(cfg.epochs):

            # will generate a random permutation of length x[0], for eg. [5,2,3,0,1,4]
            perm = torch.randperm(x_chunks.size(0))

            # mixing up x and y using same permutation
            x = x_chunks[perm]
            y = y_chunks[perm]

            i = random.randint(0, len(x) - 1 - cfg.batch_size)
            x_batch = x[i : i + cfg.batch_size]                 # [32, seq_len]
            y_batch = y[i : i + cfg.batch_size]                 # [32, seq_len]

            output, loss = self.forward(x_batch, y_batch)
            loss.backward()
            print(f"epoch: {epoch}, loss: {loss.item():.4f}")
            optimizer.step()
            optimizer.zero_grad()
