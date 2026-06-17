import torch
from model import Decoder
from tokenizer_optimized import Tokenizer

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")
checkpoint = torch.load("checkpoint.pt", weights_only=False, map_location=device)
config = checkpoint["config"]
decoder = Decoder(config)
decoder.load_state_dict(checkpoint["model_state"])
decoder.to(device)
decoder.eval()

tokenizer = Tokenizer.load("tokens.json")


def sample(logits: torch.Tensor, temperature, top_k) -> int:
    logits = logits / temperature
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k)
    logits[logits < values[-1]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def generate(
    prompt: str, max_new_tokens: int = 500, temperature: float = 0.7, top_k: int = 50
) -> str:
    tokens = tokenizer.encode(prompt)
    pad_id = tokenizer.special_tokens.get("<|padding|>", 0)

    for _ in range(max_new_tokens):
        context = tokens[-config.seq_len :]

        # pad to seq_len
        pad_len = config.seq_len - len(context)
        padded = [pad_id] * pad_len + context
        x = torch.tensor(padded).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = decoder(x)  # [1, T, vocab_size]

        next_token = sample(logits[0, -1, :], temperature=temperature, top_k=top_k)
        tokens.append(next_token)

        if next_token == tokenizer.special_tokens.get("<|endoftext|>"):
            break

    return tokenizer.decode(tokens[len(tokenizer.encode(prompt)) :])


while True:
    prompt = input("You: ")
    if prompt.lower() in ("exit", "quit"):
        break
    print("Model:", generate(prompt))
