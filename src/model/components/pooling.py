from typing import Optional
from torch import BoolTensor, nn
import torch
import torch.nn.functional as F


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
   
    def forward(self, x, mask: Optional[BoolTensor] = None):
        N, *dims, d = x.shape
        x = x.view(N, -1, d)

        if mask is not None:
            mask = mask.view(N, -1).bool()
            x = torch.masked_fill(x, ~mask[:, :, None], 0.)
            # (N x d)
            pooled = x.sum(1) / (mask.sum(1)[:, None] + 1e-7)
        else:
            pooled = torch.mean(x, dim=1)

        return pooled
        