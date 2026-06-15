from tokenizer_optimized import Tokenizer
from datasets import load_dataset
from config import ModelConfig
from model import Decoder
import torch

import truststore

truststore.inject_into_ssl()

tokenizer = Tokenizer.load("tokens.json")
ds = load_dataset("roneneldan/TinyStories", split="train[:1000]")
text = " ".join(ds["text"])
tokens = tokenizer.encode(text)

config = ModelConfig()
decoder = Decoder(config)
decoder.fit(tokens, config)
torch.save(
    {
        "model_state": decoder.state_dict(),
        "config": config,
    },
    "checkpoint.pt",
)
