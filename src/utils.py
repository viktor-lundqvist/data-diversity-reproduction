import numpy as np
import torch

def to_seq(xs, ys):
    data, targets = xs, ys
    batch_size, seq_len, n_dims = data.shape
    dtype = data.dtype
    data = torch.concatenate([torch.zeros((batch_size, seq_len, 1), dtype=dtype), data], axis=2)
    targets = torch.concatenate([targets[:, :, None], torch.zeros((batch_size, seq_len, n_dims), dtype=dtype)], axis=2)
    seq = torch.stack([data, targets], axis=2).reshape(batch_size, 2 * seq_len, n_dims + 1)
    return seq

def seq_to_targets(seq):
    return seq[:, ::2, 0]
