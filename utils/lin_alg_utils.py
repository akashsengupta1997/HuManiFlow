import torch


def batch_trace(m):
    d = m.shape[-1]
    assert m.shape[-2] == d
    i = torch.arange(d, dtype=torch.int64)
    return m[..., i, i].sum(-1)