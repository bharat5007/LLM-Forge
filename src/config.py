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


    # training parameters
    batch_size: int = 32
    epochs: int = 10000
