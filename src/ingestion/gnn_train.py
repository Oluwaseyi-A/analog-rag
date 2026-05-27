"""Contrastive GIN training on AnalogGym circuits (DICE-style InfoNCE).

Usage:
    python -m src.ingestion.gnn_train \
        --rag-data /data/rag_data \
        --epochs 50 \
        --output /app/data/gin_weights.pt
"""

import os
import sys
import json
import copy
import random
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.ingestion.netlist_parser import load_netlist_file, parse_and_canonicalize
from src.ingestion.graph_builder import build_homogeneous_graph
from src.ingestion.embedder import GINEncoder

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Augmentations ─────────────────────────────────────────────────────────────

def augment_graph(data: Data, mode: str = "node_perm") -> Data:
    """Apply a random augmentation to a homogeneous PyG graph.

    Modes:
    - node_perm: randomly permute node order (preserves topology)
    - edge_drop: drop 10% of edges randomly
    - feature_noise: add small Gaussian noise to node features
    """
    data = copy.deepcopy(data)

    if mode == "node_perm":
        n = data.x.shape[0]
        perm = torch.randperm(n)
        data.x = data.x[perm]
        if data.edge_index.shape[1] > 0:
            # Remap node indices
            inv_perm = torch.zeros(n, dtype=torch.long)
            inv_perm[perm] = torch.arange(n)
            data.edge_index = inv_perm[data.edge_index]

    elif mode == "edge_drop":
        if data.edge_index.shape[1] > 0:
            n_edges = data.edge_index.shape[1]
            keep = torch.rand(n_edges) > 0.1
            data.edge_index = data.edge_index[:, keep]

    elif mode == "feature_noise":
        noise = torch.randn_like(data.x) * 0.05
        data.x = data.x + noise

    return data


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_graphs_from_rag_data(rag_data_path: Path) -> list[tuple[str, Data]]:
    """Load all netlist_resolved.sp files and build homogeneous graphs."""
    graphs = []
    runs_dir = rag_data_path / "runs"
    for topo_dir in sorted(runs_dir.iterdir()):
        if not topo_dir.is_dir():
            continue
        for sc_dir in sorted(topo_dir.iterdir()):
            if not sc_dir.is_dir():
                continue
            for sample_dir in sorted(sc_dir.iterdir()):
                netlist_file = sample_dir / "netlist_resolved.sp"
                if not netlist_file.exists():
                    continue
                try:
                    text = load_netlist_file(netlist_file)
                    parsed = parse_and_canonicalize(text)
                    g = build_homogeneous_graph(parsed)
                    cid = f"{topo_dir.name}/{sc_dir.name}/{sample_dir.name}"
                    graphs.append((cid, g, topo_dir.name))
                except Exception as e:
                    log.warning(f"Skipping {sample_dir}: {e}")
    return graphs


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric InfoNCE contrastive loss between two sets of embeddings."""
    n = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # [2n, d]
    z = F.normalize(z, dim=1)
    sim = z @ z.T  # [2n, 2n]
    sim = sim / temperature

    # Mask out self-similarity
    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # Positive pairs: (i, i+n) and (i+n, i)
    labels = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)
    loss = F.cross_entropy(sim, labels)
    return loss


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    graphs: list,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    temperature: float = 0.1,
    hidden_dim: int = 256,
    output_path: str = "/app/data/gin_weights.pt",
    device_str: str = "cpu",
):
    device = torch.device(device_str)
    model = GINEncoder(in_channels=24, hidden_dim=hidden_dim, out_dim=hidden_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Same-topology grouping for hard negatives (within-topology = hard)
    topo_to_graphs = {}
    for cid, g, topo in graphs:
        topo_to_graphs.setdefault(topo, []).append((cid, g))

    log.info(f"Training GIN on {len(graphs)} graphs, {len(topo_to_graphs)} topologies, {epochs} epochs")

    for epoch in range(epochs):
        model.train()
        # Sample random pairs within each topology as positives
        batch_pos1, batch_pos2 = [], []
        random.shuffle(graphs)
        for cid, g, topo in graphs[:batch_size * 2]:
            if g.x.shape[0] < 2:
                continue
            aug1 = augment_graph(g, mode=random.choice(["node_perm", "edge_drop", "feature_noise"]))
            aug2 = augment_graph(g, mode=random.choice(["node_perm", "edge_drop", "feature_noise"]))
            batch_pos1.append(aug1)
            batch_pos2.append(aug2)
            if len(batch_pos1) >= batch_size:
                break

        if not batch_pos1:
            log.warning("Empty batch — skipping epoch")
            continue

        b1 = Batch.from_data_list(batch_pos1).to(device)
        b2 = Batch.from_data_list(batch_pos2).to(device)

        z1 = model(b1)
        z2 = model(b2)
        loss = info_nce_loss(z1, z2, temperature=temperature)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            log.info(f"Epoch {epoch+1}/{epochs} | loss={loss.item():.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    # Save weights
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    log.info(f"Saved GIN weights to {output_path}")
    return model


def evaluate_topology_recall(model: GINEncoder, graphs: list, k: int = 5) -> float:
    """Evaluate topology retrieval recall@k: fraction of queries where top-k includes same topology."""
    model.eval()
    # Embed all
    all_embs = []
    with torch.no_grad():
        for cid, g, topo in graphs:
            if g.x.shape[0] < 2:
                all_embs.append(torch.zeros(1, 256))
                continue
            b = Batch.from_data_list([g])
            emb = model(b)
            all_embs.append(emb)
    embs = torch.cat(all_embs, dim=0)  # [N, 256]
    topos = [t for _, _, t in graphs]

    # Compute pairwise cosine similarity
    normed = F.normalize(embs, dim=1)
    sim = normed @ normed.T

    hits = 0
    for i, topo in enumerate(topos):
        sims_i = sim[i].clone()
        sims_i[i] = -1e9  # exclude self
        top_k = sims_i.topk(k).indices.tolist()
        if any(topos[j] == topo for j in top_k):
            hits += 1

    recall = hits / len(graphs)
    log.info(f"Topology recall@{k}: {recall:.3f}")
    return recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-data", default="/data/rag_data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output", default="/app/data/gin_weights.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--weights", help="Load existing weights for eval")
    args = parser.parse_args()

    rag_data_path = Path(args.rag_data)
    graphs = load_graphs_from_rag_data(rag_data_path)
    log.info(f"Loaded {len(graphs)} graphs")

    if args.eval_only:
        model = GINEncoder(hidden_dim=args.hidden_dim)
        if args.weights:
            model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        evaluate_topology_recall(model, graphs)
        return

    model = train(
        graphs=graphs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        hidden_dim=args.hidden_dim,
        output_path=args.output,
        device_str=args.device,
    )
    evaluate_topology_recall(model, graphs)


if __name__ == "__main__":
    main()
