"""Unit tests for ResGATv2 and HydroBoostGAT model architectures."""

import pytest
import torch
from pathlib import Path
from src.models.res_gat import ResGATv2
from src.models.hydroboost import HydroBoostGAT

GRAPH_PATH = Path("data/graph/graph_knn_k7_physical.pt")


def test_graph_data_integrity():
    assert GRAPH_PATH.exists(), f"Graph file not found at {GRAPH_PATH}"
    data = torch.load(GRAPH_PATH, weights_only=False)
    assert "x" in data
    assert "edge_index" in data
    assert "edge_attr" in data
    assert "y_dried" in data
    assert data["x"].shape[0] == 5689
    assert data["edge_attr"].shape[1] == 5, "Expected standardized 5D edge tensor"


def test_resgatv2_forward_pass():
    data = torch.load(GRAPH_PATH, weights_only=False)
    x = data["x"][:100]  # Subgraph test
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_attr = torch.randn((3, 5), dtype=torch.float32)

    model = ResGATv2(in_dim=x.shape[1], hidden_dim=32, edge_dim=5, heads=2, num_layers=1)
    out = model(x, edge_index, edge_attr)
    assert out.shape == (100, 2), f"Expected shape (100, 2), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN detected in ResGATv2 output"


def test_hydroboost_gat_initialization():
    hb = HydroBoostGAT(in_dim=25, hidden_dim=64, edge_dim=5, heads=2, num_layers=1, random_state=42)
    assert hb.cb_params["random_seed"] == 42
    assert hb.gat_params["num_layers"] == 1
