from data import load_toy
import random
import torch

(train, test), (i2c, c2i) = load_toy(final=False)

print(train.shape)

#Treat this as a data loader -> call it outside anything where gradients are computed, and no required_grad=True
def batch_dataset(dataset, batch_size: int, seq_len: int):
    "return a 2D tensor of shape (Batch_size, seq_len) --> Batch rows, sequence length columns"
    N = len(dataset)
    assert seq_len <= N, "seq_len cannot be larger than dataset length"
    max_start = N - seq_len
    start_pos = torch.randint(0, max_start + 1, size=(batch_size,), dtype=torch.long)
    instances = torch.arange(seq_len, dtype=torch.long)
    batch = dataset[start_pos.unsqueeze(1) + instances.unsqueeze(0)]
    return batch



