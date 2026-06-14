import json
import regex as re
from datasets import load_dataset
import truststore

truststore.inject_into_ssl()


class Tokenizer:
    def __init__(self, special_tokens=None):
        self.merges = {}
        self.special_tokens = special_tokens or {}
        self.vocab = {}

    def stats(self, text, count=None):
        count = count if count is not None else {}
        for pair in zip(text, text[1:]):
            count[pair] = 1 + count.get(pair, 0)

    def merge(self, tokens, max_pair, new_idx):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == max_pair:
                new_tokens.append(new_idx)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def encode_chunk(self, tokens):
        while True:
            count = {}
            self.stats(tokens, count)
            if not count:
                break
            max_count = max(
                count.items(),
                key=lambda item: item[1] if item[0] in self.merges else float("-inf"),
            )
            if max_count[0] not in self.merges:
                break
            tokens = self.merge(tokens, max_count[0], self.merges[max_count[0]])
        return tokens

    def encode(self, text):
        if self.special_tokens:
            pattern = (
                "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            )
            chunks = re.split(pattern, text)
        else:
            chunks = [text]

        result = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                result.append(self.special_tokens[chunk])
            else:
                tokens = self.apply_regex(chunk)
                for token in tokens:
                    result.extend(self.encode_chunk(token))
        return result

    def decode(self, tokens):
        tokens = b"".join(self.vocab[token] for token in tokens)
        return tokens.decode("utf-8", errors="replace")

    def apply_regex(self, text):
        gpt2pat = re.compile(
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        )
        tokens = re.findall(gpt2pat, text)
        return [list(ch.encode("utf-8")) for ch in tokens]

    def training(self, text, vocab_size):
        tokens = self.apply_regex(text)
        for i in range(vocab_size - 256):
            count = {}
            for sub_text in tokens:
                self.stats(sub_text, count)
            if not count:
                print(f"No pairs left to merge after {i} steps. Stopping early.")
                break
            max_count = max(count.items(), key=lambda item: item[1])
            new_idx = 256 + i
            self.merges[max_count[0]] = new_idx
            for j in range(len(tokens)):
                tokens[j] = self.merge(tokens[j], max_count[0], new_idx)
            print(f"Epoch: {i}")

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        self.register_self_token()

    def register_self_token(self):
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode("utf-8")

    def save(self, path):
        data = {
            "merges": {f"{p0},{p1}": idx for (p0, p1), idx in self.merges.items()},
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(data["special_tokens"])
        tokenizer.merges = {
            (int(p0), int(p1)): idx
            for key, idx in data["merges"].items()
            for p0, p1 in [key.split(",")]
        }
        tokenizer.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in tokenizer.merges.items():
            tokenizer.vocab[idx] = tokenizer.vocab[p0] + tokenizer.vocab[p1]
        tokenizer.register_self_token()
        return tokenizer


# ds = load_dataset("roneneldan/TinyStories", split="train") ############## Too heavy to run locally
ds = load_dataset("roneneldan/TinyStories", split="train[:1000]")
text = " ".join(ds["text"])
special_tokens = {"<|endoftext|>": 50256, "<|padding|>": 50257}
tokenizer = Tokenizer(special_tokens)
tokenizer.training(text, 10000)
tokenizer.save("tokenizer.json")
