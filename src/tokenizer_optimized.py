import json
import multiprocessing as mp
from collections import Counter, defaultdict

import regex as re
from datasets import load_dataset

# Compile once at module level - avoids recompiling on every encode() call
GPT2_PATTERN = re.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def _process_batch(texts):
    """Worker function: pretokenize a batch of documents and return a
    Counter of unique pretokens (as tuples of byte ints) -> frequency.
    Runs in a separate process (multiprocessing)."""
    counter = Counter()
    for text in texts:
        for tok in re.findall(GPT2_PATTERN, text):
            counter[tuple(tok.encode("utf-8"))] += 1
    return counter


class Tokenizer:
    def __init__(self, special_tokens=None):
        self.merges = {}
        self.special_tokens = special_tokens or {}
        self.vocab = {}

    # ---------------------------------------------------------------
    # Encoding / Decoding
    # ---------------------------------------------------------------

    def apply_regex(self, text):
        tokens = re.findall(GPT2_PATTERN, text)
        return [list(ch.encode("utf-8")) for ch in tokens]

    def merge(self, tokens, pair, new_idx):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(new_idx)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def encode_chunk(self, tokens):
        # Apply merges in the order they were *learned* (lowest merge-index
        # first). This is the standard correct approach - merges learned
        # earlier represent more frequent/foundational pairs and must be
        # applied first for tokenization to match training.
        while len(tokens) >= 2:
            pairs_present = set(zip(tokens, tokens[1:]))
            candidates = pairs_present & self.merges.keys()
            if not candidates:
                break
            best_pair = min(candidates, key=lambda p: self.merges[p])
            tokens = self.merge(tokens, best_pair, self.merges[best_pair])
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

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    def training(self, texts, vocab_size, num_workers=None, log_every=500):
        """
        texts: list[str] - list of documents (e.g. ds["text"])
        vocab_size: target vocab size (>256). num_merges = vocab_size - 256
        """
        num_merges = vocab_size - 256
        if num_merges <= 0:
            raise ValueError("vocab_size must be > 256")

        # -----------------------------------------------------------
        # Step 1: Parallel pretokenization + dedup into unique chunks
        # -----------------------------------------------------------
        # Instead of keeping every single word occurrence (hundreds of
        # millions of tiny lists), we collapse identical pretokens
        # ("the", " the", "the.", etc.) into ONE entry + a frequency count.
        # TinyStories has a tiny vocabulary -> this turns ~500M token
        # occurrences into maybe tens/hundreds of thousands of unique chunks.
        print("Pretokenizing (multiprocessing)...")
        num_workers = num_workers or max(1, mp.cpu_count() - 1)
        batch_size = max(1, len(texts) // (num_workers * 8))
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        chunk_counter = Counter()
        with mp.Pool(num_workers) as pool:
            for partial in pool.imap_unordered(_process_batch, batches):
                chunk_counter.update(partial)

        chunk_tokens = [list(k) for k in chunk_counter.keys()]
        chunk_freqs = list(chunk_counter.values())
        print(f"Unique pretokens: {len(chunk_tokens)} "
              f"(from {sum(chunk_freqs)} total occurrences)")

        # -----------------------------------------------------------
        # Step 2: Initial pair counts + reverse index (pair -> chunks)
        # -----------------------------------------------------------
        pair_counts = defaultdict(int)
        pair_to_chunks = defaultdict(set)

        for idx, ids in enumerate(chunk_tokens):
            freq = chunk_freqs[idx]
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += freq
                pair_to_chunks[pair].add(idx)

        # -----------------------------------------------------------
        # Step 3: Merge loop with incremental (diff-based) updates
        # -----------------------------------------------------------
        self.merges = {}

        for i in range(num_merges):
            if not pair_counts:
                print(f"No pairs left to merge after {i} steps. Stopping early.")
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]
            if best_count <= 0:
                print(f"No pairs left to merge after {i} steps. Stopping early.")
                break

            new_idx = 256 + i
            self.merges[best_pair] = new_idx

            # Only chunks that actually contain best_pair need to be touched.
            affected = pair_to_chunks.get(best_pair, set())
            for chunk_idx in list(affected):
                old_ids = chunk_tokens[chunk_idx]
                freq = chunk_freqs[chunk_idx]
                if len(old_ids) < 2:
                    continue

                old_pairs = Counter(zip(old_ids, old_ids[1:]))
                new_ids = self.merge(old_ids, best_pair, new_idx)
                new_pairs = Counter(zip(new_ids, new_ids[1:]))

                # Remove counts for pairs that existed before the merge
                for p, c in old_pairs.items():
                    pair_counts[p] -= c * freq

                # Add counts for pairs that exist after the merge
                for p, c in new_pairs.items():
                    pair_counts[p] += c * freq
                    pair_to_chunks[p].add(chunk_idx)

                chunk_tokens[chunk_idx] = new_ids

            pair_counts[best_pair] = 0
            pair_to_chunks.pop(best_pair, None)

            # Periodic cleanup so pair_counts/pair_to_chunks don't grow
            # unbounded with stale zero entries
            if (i + 1) % 1000 == 0:
                pair_counts = defaultdict(
                    int, {p: c for p, c in pair_counts.items() if c > 0}
                )

            if i % log_every == 0:
                print(f"merge {i}/{num_merges}: {best_pair} -> {new_idx} "
                      f"(count={best_count})")

        # -----------------------------------------------------------
        # Step 4: Build vocab
        # -----------------------------------------------------------
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        self.register_self_token()

    def register_self_token(self):
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode("utf-8")

    # ---------------------------------------------------------------
    # Save / Load
    # ---------------------------------------------------------------

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


if __name__ == "__main__":
    ds = load_dataset("roneneldan/TinyStories", split="train")
    special_tokens = {"<|endoftext|>": 50256, "<|padding|>": 50257}
    tokenizer = Tokenizer(special_tokens)
    tokenizer.training(ds["text"], 50000)
    tokenizer.save("tokenizer.json")