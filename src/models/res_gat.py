"""
ResGATv2: shallow GATv2 with an ego-node linear skip connection.

    h_i = GELU( W_ego x_i + Dropout( GATv2Conv(x, E, e)_i ) )

Default: 1 message-passing layer (receptive field = k-NN neighbourhood, mean
edge length ~262 m), edge attributes from ``src.graph_builder``, no
normalisation layer. The flags ``ego_skip``, ``layer_norm`` and ``num_layers``
exist so that every design choice is tested as an ablation in
``src.benchmark`` rather than asserted.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class ResGATv2(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, edge_dim: int = 5, heads: int = 2,
                 num_layers: int = 1, dropout: float = 0.15, num_classes: int = 2,
                 ego_skip: bool = True, layer_norm: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.ego_skip = ego_skip
        self.dropout = nn.Dropout(dropout)
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList()
        self.egos = nn.ModuleList()
        self.norms = nn.ModuleList()
        for l in range(num_layers):
            self.convs.append(GATv2Conv(dims[l], hidden_dim // heads, heads=heads, edge_dim=edge_dim,
                                        concat=True, dropout=dropout, add_self_loops=True))
            self.egos.append(nn.Linear(dims[l], hidden_dim) if ego_skip else nn.Identity())
            self.norms.append(nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity())
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=False):
        h = x
        atts = []
        for conv, ego, norm in zip(self.convs, self.egos, self.norms):
            if return_attention_weights:
                hc, att = conv(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)
                atts.append(att)
            else:
                hc = conv(h, edge_index, edge_attr=edge_attr)
            z = self.dropout(hc)
            if self.ego_skip:
                z = ego(h) + z
            h = F.gelu(norm(z))
        out = self.classifier(h)
        if return_attention_weights:
            return out, (atts[0] if len(atts) == 1 else tuple(atts))
        return out
