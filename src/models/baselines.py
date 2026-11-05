"""
Baselines.

Tabular (graph-free): logistic regression, random forest, XGBoost, CatBoost.
Neural, graph-free: MLP (same training loop as the GNNs, so any GNN gain over
the MLP is attributable to message passing rather than to the optimiser).
Spatial interpolation: distance-weighted k-NN vote of *labelled* neighbours --
the obvious baseline for any method that exploits label autocorrelation.
GNNs: GCN (Kipf & Welling, 2017), GraphSAGE (Hamilton et al., 2017),
GATv2 (Brody et al., 2022) without skip connections.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
from xgboost import XGBClassifier


def get_tabular_models(seed: int = 42, pos_weight: float = 1.0):
    return {
        "Logistic Regression": lambda: LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=seed
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05, scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0,
        ),
        "CatBoost": lambda: CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05, auto_class_weights="Balanced",
            verbose=0, random_seed=seed, thread_count=-1, allow_writing_files=False,
        ),
    }


def knn_label_predict(coords_m: np.ndarray, y: np.ndarray, labelled: np.ndarray, k: int = 7) -> np.ndarray:
    """Inverse-distance-weighted mean label of the k nearest labelled springs, for every node."""
    idx_l = np.where(labelled)[0]
    d, j = cKDTree(coords_m[idx_l]).query(coords_m, k=k)
    w = 1.0 / np.maximum(d, 1.0)
    return (w * y[idx_l][j]).sum(1) / w.sum(1)


class MLP(nn.Module):
    """Graph-free 2-layer perceptron with the same width/dropout as the GNNs."""
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, edge_index=None, edge_attr=None):
        return self.net(x)


class SpringGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.25):
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
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.25):
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
    """2-layer GATv2 with edge attributes, no skip connections or normalisation."""
    def __init__(self, in_dim: int, hidden_dim: int = 64, edge_dim: int = 5, heads: int = 2, dropout: float = 0.25):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=1, edge_dim=edge_dim, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.dropout(F.elu(self.conv1(x, edge_index, edge_attr=edge_attr)))
        h = self.dropout(F.elu(self.conv2(h, edge_index, edge_attr=edge_attr)))
        return self.classifier(h)
