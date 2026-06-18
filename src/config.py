from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Vocab and Sequence
    vocab_size: int = 50002  # vocab size
    seq_len: int = 256  # length of sequence to be fed

    # Architecture
    heads_num: int = 4  # number of heads in each attentition
    decoder_num: int = 10  # number of decoders
    emb_size: int = 576  # embedding size
    kv_heads: int = 3  # number of kv heads
    q_heads: int = 9  # number of q heads
    heads_emb: int = 0  # head embedding size

    # training parameters
    batch_size: int = 32
    epochs: int = 30000

    def __post_init__(self):
        if self.q_heads % self.kv_heads != 0:
            raise ValueError(
                f"q_heads ({self.q_heads}) must be divisible by kv_heads ({self.kv_heads})"
            )

        if self.emb_size % self.q_heads != 0:
            raise ValueError(
                f"emb_size {self.emb_size} needs to be divisible by {self.q_heads}"
            )

        self.heads_emb = self.emb_size // self.q_heads
