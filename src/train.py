from tokenizer_optimized import Tokenizer
from datasets import load_dataset
from config import ModelConfig
from model import Decoder
import torch

import truststore

truststore.inject_into_ssl()
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

tokenizer = Tokenizer.load("tokens.json")
ds = load_dataset("roneneldan/TinyStories", split="train[:1000]")
text = " ".join(ds["text"])
tokens = tokenizer.encode(text)

config = ModelConfig()
decoder = Decoder(config).to(device)
decoder.fit(tokens, config)
torch.save(
    {
        "model_state": decoder.state_dict(),
        "config": config,
    },
    "checkpoint.pt",
)
