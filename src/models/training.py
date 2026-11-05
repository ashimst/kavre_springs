"""Shared full-batch training loop for all neural models (MLP and GNNs)."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_node_classifier(model_factory, x, edge_index, edge_attr, y, train_mask, val_mask, seed: int,
                          max_epochs: int = 300, patience: int = 40, lr: float = 3e-3,
                          weight_decay: float = 1e-4):
    """Train with class-balanced cross-entropy; early-stop on validation ROC-AUC.

    Returns (probabilities for all nodes at the best validation epoch, the trained model).
    """
    set_seed(seed)
    model = model_factory().to(DEVICE)
    x_d = x.to(DEVICE)
    ei_d = edge_index.to(DEVICE) if edge_index is not None else None
    ea_d = edge_attr.to(DEVICE) if edge_attr is not None else None
    y_d = y.to(DEVICE)
    y_np = y.cpu().numpy()
    tr = torch.as_tensor(train_mask, device=DEVICE)

    n_pos = float(y_np[train_mask].sum())
    n_neg = float(train_mask.sum() - n_pos)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, n_neg / max(n_pos, 1.0)], device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc, best_state, best_probs, since_best = -1.0, None, None, 0
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(x_d, ei_d, ea_d)[tr], y_d[tr])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            p = F.softmax(model(x_d, ei_d, ea_d), dim=-1)[:, 1].cpu().numpy()
        auc = roc_auc_score(y_np[val_mask], p[val_mask])
        if auc > best_auc:
            best_auc, best_probs, since_best = auc, p, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return best_probs, model
