"""Embed netlists (text) and circuit graphs (GNN) for vector store insertion."""

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool
from openai import OpenAI


# ── GIN Encoder ──────────────────────────────────────────────────────────────

class GINEncoder(nn.Module):
    """4-layer GIN with mean+max pooling → 256-dim graph embedding."""

    def __init__(self, in_channels: int = 24, hidden_dim: int = 256, out_dim: int = 256, n_layers: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dims = [in_channels] + [hidden_dim] * n_layers
        for i in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(dims[i], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.proj = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.shape[0], dtype=torch.long)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        # Pool only over device nodes (first n_devices nodes per graph)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        out = torch.cat([mean_pool, max_pool], dim=1)
        return F.normalize(self.proj(out), p=2, dim=-1)


_gin_model: Optional[GINEncoder] = None


def _get_gin(weights_path: Optional[str] = None, device: str = "cpu") -> GINEncoder:
    global _gin_model
    if _gin_model is None:
        _gin_model = GINEncoder().to(device)
        _gin_model.eval()
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=device)
            _gin_model.load_state_dict(state)
    return _gin_model


def embed_graph(pyg_data: Data, weights_path: Optional[str] = None) -> list[float]:
    """Encode a homogeneous PyG Data object → 256-dim embedding list."""
    model = _get_gin(weights_path)
    with torch.no_grad():
        if pyg_data.x.shape[0] == 0:
            return [0.0] * 256
        emb = model(pyg_data)
    return emb.squeeze(0).tolist()


# ── Text Embedder ─────────────────────────────────────────────────────────────

_openai_client: Optional[OpenAI] = None


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def embed_text(text: str, model: str = "text-embedding-3-large") -> list[float]:
    """Embed a string with OpenAI → 3072-dim vector."""
    client = _get_openai()
    response = client.embeddings.create(input=[text[:8000]], model=model)
    return response.data[0].embedding


def embed_texts_batch(texts: list[str], model: str = "text-embedding-3-large", batch_size: int = 16) -> list[list[float]]:
    """Embed a list of strings in batches."""
    client = _get_openai()
    results = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:8000] for t in texts[i:i + batch_size]]
        resp = client.embeddings.create(input=batch, model=model)
        results.extend([r.embedding for r in sorted(resp.data, key=lambda x: x.index)])
    return results


# ── Behavior Feature Vector ───────────────────────────────────────────────────

BEHAVIOR_FIELDS = [
    "measured__dcgain", "measured__gbp", "measured__phase_in_deg",
    "measured__SR", "measured__power", "measured__area",
    "measured__cmrrdc", "measured__dcpsrn", "measured__dcpsrp",
    "measured__foml", "measured__foms", "measured__settlingTime",
    "measured__d_settle", "measured__vos25", "measured__stable",
]

BEHAVIOR_NORMS = {
    "measured__dcgain": (0, 120),
    "measured__gbp": (0, 50e6),
    "measured__phase_in_deg": (-180, 180),
    "measured__SR": (-10, 100),
    "measured__power": (0, 5e-3),
    "measured__area": (0, 5000),
    "measured__cmrrdc": (-120, 0),
    "measured__dcpsrn": (-120, 0),
    "measured__dcpsrp": (-120, 0),
    "measured__foml": (0, 500),
    "measured__foms": (0, 1e9),
    "measured__settlingTime": (0, 1e-4),
    "measured__d_settle": (0, 1),
    "measured__vos25": (-1e-2, 1e-2),
    "measured__stable": (0, 1),
}


def build_behavior_vector(result_row: dict, dim: int = 64) -> list[float]:
    """Build a normalized 64-dim behavior feature vector from result_row.json."""
    raw = []
    for field in BEHAVIOR_FIELDS:
        val = result_row.get(field, 0.0)
        if val is None:
            val = 0.0
        lo, hi = BEHAVIOR_NORMS.get(field, (0, 1))
        span = hi - lo
        if span == 0:
            norm = 0.0
        else:
            norm = max(0.0, min(1.0, (float(val) - lo) / span))
        raw.append(norm)

    # Pad or truncate to dim
    if len(raw) < dim:
        raw.extend([0.0] * (dim - len(raw)))
    return raw[:dim]


def build_behavior_embedding(result_row: dict, dim: int = 64) -> list[float]:
    """Return the 64-dim behavior vector (acts as ChromaDB embedding for analog_behavior)."""
    return build_behavior_vector(result_row, dim)
