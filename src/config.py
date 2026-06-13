from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Vocab and Sequence
    vocab_size: int = 50002  # vocab size
    seq_len: int = 1024  # length of sequence to be fed

    # Architecture
    heads_num: int = 4  # number of heads in each attentition
    decoder_num: int = 6  # number of decoders
    emb_size: int = 512  # embedding size
    heads_emb: int = 64
    kv_heads: int = 4
    q_heads: int = 16

    # training parameters
    batch_size: int = 32
    epochs: int = 10000
