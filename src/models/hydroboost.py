"""
Flagship Architecture: HydroBoost-GAT.

Dual-Stage Hybrid Learning Paradigm:
1. Stage 1 (Tabular Non-Linear Priors):
   - Gradient-boosted symmetric decision trees (CatBoost) capture sharp, non-linear,
     discontinuous tabular thresholds (e.g., 2015 earthquake damage, annual precipitation bounds).
2. Stage 2 (Physical Anisotropic Graph Attention):
   - Lightweight ResGATv2 (1-2 layer ego-skip) conditions message passing on standardized 5D physical edges
     e_ij = [log1p(d_ij), unit_dx, unit_dy, delta_z, slope_gradient].
3. Stage 3 (Topological Spatial Residual Correction):
   - Diffuses residual error e_i = y_i - p_i along subterranean fracture conduits
     using the normalized hydro-topographic graph Laplacian.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from torch_geometric.utils import to_dense_adj
from src.models.res_gat import ResGATv2


class HydroBoostGAT:
    def __init__(
        self,
        in_dim: int = 25,
        hidden_dim: int = 64,
        edge_dim: int = 5,
        heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.15,
        catboost_iterations: int = 400,
        fusion_weight: float = 0.60,
        diffusion_alpha: float = 0.40,
        diffusion_steps: int = 3,
        random_state: int = 42,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fusion_weight = fusion_weight
        self.diffusion_alpha = diffusion_alpha
        self.diffusion_steps = diffusion_steps
        self.random_state = random_state
        
        self.cb_params = {
            "iterations": catboost_iterations,
            "depth": 6,
            "learning_rate": 0.05,
            "auto_class_weights": "Balanced",
            "verbose": 0,
            "random_seed": random_state,
        }
        
        self.gat_params = {
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "edge_dim": edge_dim,
            "heads": heads,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        
        self.cb_model = None
        self.gat_model = None

    def fit_and_predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        y: torch.Tensor,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        test_mask: np.ndarray,
        epochs: int = 100,
        lr: float = 0.003,
    ) -> np.ndarray:
        """Fit HydroBoost-GAT on training fold and return test probabilities."""
        x_np = x.numpy()
        y_np = y.numpy()
        num_nodes = len(y_np)
        
        # 1. Stage 1: CatBoost Tabular Prior
        self.cb_model = CatBoostClassifier(**self.cb_params)
        self.cb_model.fit(x_np[train_mask], y_np[train_mask])
        p_cb_all = self.cb_model.predict_proba(x_np)[:, 1]
        
        # 2. Stage 2: ResGATv2 Graph Attention
        x_dev = x.to(self.device)
        edge_idx_dev = edge_index.to(self.device)
        edge_att_dev = edge_attr.to(self.device) if edge_attr is not None else None
        y_dev = y.to(self.device)
        
        tr_t = torch.tensor(train_mask, device=self.device)
        va_t = torch.tensor(val_mask, device=self.device)
        te_t = torch.tensor(test_mask, device=self.device)
        
        torch.manual_seed(self.random_state)
        self.gat_model = ResGATv2(**self.gat_params).to(self.device)
        optimizer = torch.optim.AdamW(self.gat_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.74], device=self.device))
        
        best_val_score = 0.0
        best_p_gat_all = None
        
        for ep in range(epochs):
            self.gat_model.train()
            optimizer.zero_grad()
            out = self.gat_model(x_dev, edge_idx_dev, edge_att_dev)
            loss = criterion(out[tr_t], y_dev[tr_t])
            loss.backward()
            optimizer.step()
            
            self.gat_model.eval()
            with torch.no_grad():
                out_all = self.gat_model(x_dev, edge_idx_dev, edge_att_dev)
                p_gat_val = F.softmax(out_all[va_t], dim=-1)[:, 1].cpu().numpy()
                from sklearn.metrics import roc_auc_score
                v_auc = roc_auc_score(y_np[val_mask], p_gat_val)
                if v_auc > best_val_score:
                    best_val_score = v_auc
                    best_p_gat_all = F.softmax(out_all, dim=-1)[:, 1].cpu().numpy()
                    
        if best_p_gat_all is None:
            best_p_gat_all = p_cb_all
            
        # 3. Stage 3: Multi-Scale Residual Spatial Diffusion
        # Formulate base hybrid prediction
        p_base = self.fusion_weight * p_cb_all + (1.0 - self.fusion_weight) * best_p_gat_all
        
        # Build normalized spatial transition matrix S
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].numpy()
        np.fill_diagonal(adj, 1.0)
        deg = adj.sum(axis=1)
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
        S = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
        
        # Training residuals
        error = np.zeros(num_nodes, dtype=np.float32)
        error[train_mask] = y_np[train_mask] - p_base[train_mask]
        
        E = error.copy()
        for _ in range(self.diffusion_steps):
            E = (1.0 - self.diffusion_alpha) * error + self.diffusion_alpha * S.dot(E)
            
        p_corrected = np.clip(p_base + 0.5 * E, 0.0, 1.0)
        return p_corrected[test_mask]

