"""
Baseline Model Architectures for Kavre Springs Desiccation Prediction.

Tabular:
1. Random Forest (Balanced trees)
2. XGBoost (Histogram gradient boosting)
3. CatBoost (Symmetric decision trees)

Graph Neural Networks:
1. SpringGCN (Kipf & Welling 2017)
2. SpringGraphSAGE (Hamilton et al. 2017)
3. VanillaGATv2 (Brody et al. 2022) - No residual connections, standard isotropic message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# =============================================================================
# Tabular Baseline Factories
# =============================================================================

def get_tabular_models(seed: int = 42):
    return {
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05, scale_pos_weight=4168/1521, eval_metric="logloss", random_state=seed, n_jobs=-1
        ),
        "CatBoost": lambda: CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05, auto_class_weights="Balanced", verbose=0, random_seed=seed
        ),
    }


# =============================================================================
# Standard GNN Baselines
# =============================================================================

class SpringGCN(nn.Module):
    """Isotropic Graph Convolutional Network."""
    def __init__(self, in_dim: int = 25, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.dropout(F.relu(self.conv1(x, edge_index)))
        h = self.dropout(F.relu(self.conv2(h, edge_index)))
        return self.classifier(h)


class SpringGraphSAGE(nn.Module):
    """Neighborhood Mean Aggregation Network."""
    def __init__(self, in_dim: int = 25, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.dropout(F.relu(self.conv1(x, edge_index)))
        h = self.dropout(F.relu(self.conv2(h, edge_index)))
        return self.classifier(h)


class VanillaGATv2(nn.Module):
    """Standard GATv2 without skip connections or normalization."""
    def __init__(self, in_dim: int = 25, hidden_dim: int = 64, edge_dim: int = 5, heads: int = 2, dropout: float = 0.25):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=1, edge_dim=edge_dim, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.dropout(F.elu(self.conv1(x, edge_index, edge_attr=edge_attr)))
        h = self.dropout(F.elu(self.conv2(h, edge_index, edge_attr=edge_attr)))
        return self.classifier(h)

