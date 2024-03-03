from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.model_utils import get_activation

class MLP(nn.Module):
    def __init__(
        self, 
        n_layers: int, 
        d_in: int, d_out: Optional[int] = None,
        use_bn=False, dropout=0.0, dropout_last_layer=True, act=nn.ReLU, d_hidden_factor: int = 4, d_hidden: Optional[int] = None):
        """
        :param num_layers: If num_hidden_layers == 0, then only use identity, for num_hidden_layers == 1, then only use linear
        :param d_in:
        :param d_hidden:
        :param d_out:
        :param use_bn
        """
        super(MLP, self).__init__()
        if act is None:
            act = nn.ReLU
        act = get_activation(act)

        if d_out is None:
            d_out = d_in
        if d_hidden is None:
            d_hidden = d_hidden_factor * d_out
        assert n_layers >= 0
        if n_layers == 0:
            assert d_in == d_out, f'If n_layers == 0, then d_in == d_out, but got {d_in} != {d_out}'
            self.layers = nn.Identity()
        else:
            current_dim_in = d_in
            layers = []
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(current_dim_in, d_hidden, bias=not use_bn))
                if use_bn:
                    layers.append(nn.BatchNorm1d(d_hidden))
                layers.append(act)
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                current_dim_in = d_hidden
            layers.append(nn.Linear(current_dim_in, d_out))
            if dropout_last_layer and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *dims, d = x.shape
        if len(dims) > 1:
            x = x.reshape(-1, d)

        x = self.layers(x)
        return x.view(*dims, -1)
