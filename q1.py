from data import load_imdb
from typing import List
import torch

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

def pad_batch(batch_sequences: List[List[int]], pad_idx: int) -> torch.LongTensor:
    """
    Pad a list of sequences (lists of ints) to the same length with pad_idx.

    Args:
    batch_sequences: list of lists, each inner list is a sequence of token indices
    pad_idx: integer index used for padding (from w2i['.pad'])

    Returns:
    Tensor of shape (batch_size, max_len) with dtype torch.long
    """
    #Find the maximum length in this batch
    max_len = max(len(seq) for seq in batch_sequences)

    # Create a list of lists all equal length by padding with pad_idx
    padded = []

    for seq in batch_sequences:
        # copy the sequence and extend with pad_idx to max_len
        padded_seq = seq + [pad_idx] * (max_len - len(seq))
        padded.append(padded_seq)

    # Convert to a torch tensor of dtype long
    batch_tensor = torch.tensor(padded, dtype=torch.long)

    return batch_tensor
