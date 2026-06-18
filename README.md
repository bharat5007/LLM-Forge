# GPT-Style Decoder-Only Transformer — Built from Scratch

A decoder-only transformer language model implemented in pure PyTorch, with no `nn.Transformer`, no HuggingFace model classes, and no pre-built attention layers. Every component — attention, normalization, positional encoding, feedforward, and the BPE tokenizer — is written from scratch. Trained on the TinyStories dataset as a demonstration of end-to-end understanding of modern LLM architecture for ML/NLP engineering roles.

---

## Architecture

The decoder block follows the LLaMA-style pattern (pre-norm, RoPE, GQA, SwiGLU) rather than the original 2017 "Attention is All You Need" design. Each of the 6 decoder layers consists of:

**Grouped Query Attention (GQA)** — `src/model.py:Head`
- 16 query heads, 4 KV heads (4:1 ratio), each head dim = 16
- K and V projections are smaller (`emb_size → head_emb × kv_heads`), then expanded via `repeat_interleave` before the attention dot-product
- Reduces KV cache size vs. full multi-head attention with no loss in expressivity at this scale

**RoPE (Rotary Position Embedding)** — `src/components.py:RoPE`
- Precomputes sin/cos buffers once via `torch.outer(positions, inv_freq)` and registers them as buffers
- Applied per-head to Q and K (not V) using pairwise rotation of even/odd dimension pairs
- The commented-out original O(B·T·C²) loop implementation is preserved in `components.py` to show the design iteration

**RMSNorm** — `src/components.py:RMSNorm`
- Pre-norm placement (normalize before attention/FFN, add residual after)
- No centering (no mean subtraction), just root-mean-square rescaling with a learned gamma

**SwiGLU feedforward** — `src/model.py:SwiGLU`
- Three projections: `gate_proj`, `up_proj`, `down_proj`
- Activation: `SiLU(gate_proj(x)) * up_proj(x)`, projected back down
- Hidden dim = 4× embedding dim (1024)

### Config (`src/config.py`)

| Parameter | Value |
|-----------|-------|
| `vocab_size` | 50002 |
| `seq_len` | 256 |
| `emb_size` | 576 |
| `decoder_num` | 10 |
| `q_heads` | 9 |
| `kv_heads` | 3 |
| `heads_emb` (derived) | 64 |
| `batch_size` | 32 |
| `epochs` (steps) | 30000 |

### Parameter count

Computed directly from the config above:

| Component | Params |
|-----------|--------|
| Token embedding (`vocab_size + 2, emb_size`) | 28,802,304 |
| Positional encoding (learned, `seq_len × emb_size`) | 147,456 |
| 10 × decoder layer (attn + FFN each) | 48,739,200 |
| Final RMSNorm | 576 |
| LM head (`Linear(emb_size, vocab_size)`) | 28,851,154 |
| **Total** | **~106.5M** |

54% of parameters are in the embedding table and LM head, with the remaining 46% (~48.7M) in the transformer layers themselves — a much healthier ratio than the smaller 31M config where the vocab layers dominated at 82%. Tied embeddings would still reduce this further and are flagged as future work.

---

## Tokenizer

Custom BPE tokenizer implemented from scratch in `src/tokenizer.py` and an optimized version in `src/tokenizer_optimized.py`. Neither uses tiktoken or HuggingFace tokenizers.

**Pretokenization**: GPT-2 regex pattern, compiled once at module level:
```python
GPT2_PATTERN = re.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
```
This splits text into word-like units before BPE, preventing merges across word boundaries.

**BPE merge training** (`tokenizer_optimized.py`): starts from 256 byte tokens, learns 49,744 merges to reach vocab size 50,000, plus 2 special tokens (`<|endoftext|>` → 50000, `<|padding|>` → 50001) for a final vocab of 50,002.

The basic `tokenizer.py` rescans all chunks for pair counts on every merge step — O(vocab_merges × corpus_size). `tokenizer_optimized.py` fixes this with two key changes:

1. **Parallel pretokenization**: uses `multiprocessing.Pool` to pretokenize documents across CPU cores, then merges `Counter` results. For TinyStories this collapses millions of word occurrences into tens of thousands of unique pretokens.

2. **Incremental pair-count updates**: maintains a `pair_counts` dict and a `pair_to_chunks` reverse index. After each merge, only the affected chunks are updated (diff of old pairs vs. new pairs), avoiding a full corpus rescan per merge step.

The trained tokenizer is saved/loaded as JSON (`tokens.json`).

---

## Training

**Dataset**: `roneneldan/TinyStories` from HuggingFace Datasets, 100,000 stories.  
**Hardware**: Google Colab, NVIDIA L4 GPU.

Training text is encoded with the pretrained tokenizer, padded to a multiple of `seq_len + 1`, then sliced into `(x, y)` chunk pairs for next-token prediction.

**Training loop** (`src/model.py:Decoder.fit`):
- Autoregressive next-token prediction, cross-entropy loss over all positions
- Each "epoch" is one randomly sampled mini-batch — 30,000 steps total, not 30,000 full passes
- AdamW, lr=1e-3, no scheduler
- Checkpoint saved to `checkpoint.pt` every 100 steps

---

## Scaling Experiment

Two model sizes were trained on the same dataset and compared qualitatively:

| Config | `emb_size` | `decoder_num` | `q_heads` / `kv_heads` | `seq_len` | Params |
|--------|-----------|--------------|------------------------|-----------|--------|
| Small  | 256       | 6            | 16 / 4                 | 128       | ~31M   |
| Large  | 576       | 10           | 9 / 3                  | 256       | ~107M  |

The large model shows clear qualitative improvements over the small one at the same number of training steps:

- **Narrative structure**: the small model often restarts mid-generation (repeating "Once upon a time" partway through), while the large model maintains a single narrative arc across multiple paragraphs.
- **Grammar and coherence**: sentences in the large model's output use correct cause-and-effect structure ("She was scared, but she remembered the words and decided to keep going") and consistent character naming across paragraphs — neither of which appeared reliably in the small model.
- **Context tracking**: the large model's longer `seq_len` (256 vs 128) lets it carry character state further, which visibly reduces contradictions within a single story.

These differences appear to come from both the larger model capacity and the doubled sequence length, though they are not ablated separately.

---

## Debugging: When Loss Decreases but the Model is Broken

During development, there was a one-line bug in the causal masking code inside `Head.forward`:

```python
# Bug (what was written):
logits = torch.tril(torch.ones(T, T, device=logits.device))

# Fix:
mask = torch.tril(torch.ones(T, T, device=logits.device))
logits = logits.masked_fill(mask == 0, float("-inf"))
```

The bug replaced the Q·K attention logits with the mask itself — a lower-triangular matrix of 0s and 1s. The model still produced a valid causal distribution after softmax (zeros become uniform, ones become higher probability), but the distribution was **completely independent of Q and K content**. Attention weights depended only on position, not on what any token actually was.

The model still trained. Loss still decreased. The output still looked like text. But the attention mechanism — the core mechanism of the architecture — was doing nothing. What the model actually learned was a position-biased unigram distribution layered on top of the FFN blocks, not contextual attention.

After fixing this one line, output shifted from incoherent high-frequency word repetition to grammatically correct, multi-sentence narrative text.

**The lesson**: training loss is a weak signal for whether a mechanism is actually functioning. A model can converge while one of its core operations is silently replaced with noise or a degenerate substitute. The right check here would have been to inspect attention weight matrices directly — a broken attention head has weights that look identical regardless of input content.

---

## Known Limitations

- **Small training slice**: even the larger run uses 100k TinyStories stories, a subset of the full dataset. The model's vocabulary of narrative patterns is narrow — it reliably produces child-like short stories but doesn't generalize beyond that domain.

- **Weak prompt conditioning**: the model tends to default to dominant learned patterns (a small set of recurring character archetypes and story structures) rather than following unusual or out-of-distribution prompts. This is a data scale and diversity limitation, not an architecture bug.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt
```

**Train the tokenizer** (only needed if you don't have `tokens.json`):
```bash
cd src
python tokenizer_optimized.py
```

**Train the model**:
```bash
cd src
python train.py
```
Saves `checkpoint.pt` every 100 steps. Edit `train.py` line 20 to change the dataset split (`train[:1000]` → `train[:100000]` for a larger run).

**Interactive inference**:
```bash
cd src
python load_model.py
```
Loads `checkpoint.pt` and `tokens.json`, then prompts for input in a loop. Type `exit` or `quit` to stop.

Generation uses top-k sampling (k=50, temperature=0.7) and stops at `<|endoftext|>` or `max_new_tokens=500`.

---

## Possible Future Work

- Scale model dimension and depth; add tied embeddings to reduce parameter count in vocab layers
- Train on the full TinyStories corpus (or a larger, more diverse dataset)
- Add top-p (nucleus) sampling and repetition penalty to `generate()`
- Fix left-padding for inference so prompt conditioning works correctly for short prompts
- KV cache for efficient autoregressive generation (currently re-computes full attention every step)
