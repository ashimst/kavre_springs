"""
Lean Residual Graph Attention Network (ResGATv2).

Physically grounded in sub-catchment hydrology:
1. Shallow Receptive Field (1-2 layers): Matches the ~315m-500m Moran's I spatial autocorrelation peak.
2. Ego-Node Linear Skip Connections: h_i = W_ego(x_i) + GATv2Conv(x, edge_index, edge_attr)
3. Zero LayerNorm: Preserves feature scale and logit magnitudes across channels.
4. Conditioned on 5-Dimensional Standardized Physical Edge Tensor (Zero Administrative Leakage).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class ResGATv2(nn.Module):
    def __init__(
        self,
        in_dim: int = 25,
        hidden_dim: int = 64,
        edge_dim: int = 5,
        heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.15,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        # Ego linear projection to match latent dimension
        self.ego_proj = nn.Linear(in_dim, hidden_dim)
        
        # Message passing layer(s)
        self.conv1 = GATv2Conv(
            in_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )
        
        if num_layers > 1:
            self.ego_proj2 = nn.Linear(hidden_dim, hidden_dim)
            self.conv2 = GATv2Conv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                edge_dim=edge_dim,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
            )
            
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            self.dropout,
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=False):
        # Layer 1: Separate ego projection + GAT message passing (No LayerNorm)
        h_ego = self.ego_proj(x)
        
        if return_attention_weights:
            h_conv, att = self.conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            h = F.gelu(h_ego + self.dropout(h_conv))
            if self.num_layers > 1:
                h_ego2 = self.ego_proj2(h)
                h_conv2, att2 = self.conv2(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)
                h = F.gelu(h_ego2 + self.dropout(h_conv2))
                return self.classifier(h), (att, att2)
            return self.classifier(h), att
        else:
            h_conv = self.conv1(x, edge_index, edge_attr=edge_attr)
            h = F.gelu(h_ego + self.dropout(h_conv))
            if self.num_layers > 1:
                h_ego2 = self.ego_proj2(h)
                h_conv2 = self.conv2(h, edge_index, edge_attr=edge_attr)
                h = F.gelu(h_ego2 + self.dropout(h_conv2))
            return self.classifier(h)

